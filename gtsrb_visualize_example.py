#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import os
import time

import numpy as np
import random
from tensorflow import set_random_seed
random.seed(123)
np.random.seed(123)
set_random_seed(123)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from visualizer import Visualizer

import utils_backdoor


##############################
#        PARAMETERS          #
##############################

DEVICE = '0'  # specify which GPU to use

DATA_DIR = 'data'  # data folder
DATA_FILE = 'gtsrb_dataset.h5'  # dataset file
MODEL_DIR = 'models'  # model directory
MODEL_FILENAME = 'gtsrb_bottom_right_white_4_target_33.h5'  # model file
RESULT_DIR = 'results'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png'

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 43  # total number of classes in the model
Y_TARGET = 33  # infected target label, used for prioritizing label scanning

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization

LR = 0.1  # learning rate
STEPS = 1000  # total optimization iterations
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE / BATCH_SIZE  # mini batch size used for early stop
INIT_COST = 1e-3  # initial weight used for balancing two objectives

# preprocessing method for the task, GTSRB uses raw pixel intensities
INTENSITY_RANGE = 'raw'
REGULARIZATION = 'l1'  # reg term to control the mask's norm
# attack success threshold of the reversed attack
ATTACK_SUCC_THRESHOLD = 0.99
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop

# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)

# parameters of the injected trigger
# this is NOT used during optimization
# start inclusive, end exclusive
PATTERN_START_ROW, PATTERN_END_ROW = 27, 31
PATTERN_START_COL, PATTERN_END_COL = 27, 31
PATTERN_COLOR = (255.0, 255.0, 255.0)
PATTERN_LIST = [
    (row_idx, col_idx, PATTERN_COLOR)
    for row_idx in range(PATTERN_START_ROW, PATTERN_END_ROW)
    for col_idx in range(PATTERN_START_COL, PATTERN_END_COL)
]

##############################
#      END PARAMETERS        #
##############################


def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):

    dataset = utils_keras.load_dataset(data_file)

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    print('X_train shape %s' % str(X_train.shape))
    print('Y_train shape %s' % str(Y_train.shape))
    print('X_test shape %s' % str(X_test.shape))
    print('Y_test shape %s' % str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test


def build_data_loader(X_train, Y_train, X_test, Y_test):

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(
        X_train, Y_train, batch_size=BATCH_SIZE)

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(
        X_test, Y_test, batch_size=BATCH_SIZE)

    return train_generator, test_generator


def gen_corner_mask():

    mask = np.zeros(MASK_SHAPE)
    centroid = (MASK_SHAPE - 1.0) / 2.0
    for row_idx in range(MASK_SHAPE[0]):
        for col_idx in range(MASK_SHAPE[1]):
            mask[row_idx, col_idx] = (
                1.0 *
                (np.square(centroid[0] - row_idx) +
                 np.square(centroid[1] - col_idx)) /
                np.sum(np.square(centroid)))
    mask = np.clip(mask, 0, 1)

    return mask


def visualize_trigger_w_mask(visualizer, gen, y_target,
                             save_pattern_flag=True):

    visualize_start_time = time.time()

    # random mask
    pattern = np.random.random(INPUT_SHAPE)
    pattern = (np.ones_like(pattern) - pattern * 1.0) * 255.0
    mask = np.random.random(MASK_SHAPE)
    mask = np.ones_like(mask) * 1.0 - mask * 1.0
    # mask = gen_corner_mask()

    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()

    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask_upsample, y_target)

    return pattern, mask_upsample, logs


def save_pattern(pattern, mask, y_target):

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('pattern', y_target)))
    utils_keras.dump_image(pattern, img_filename, 'png')

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('mask', y_target)))
    utils_keras.dump_image(np.expand_dims(mask, axis=2) * 255,
                           img_filename,
                           'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('fusion', y_target)))
    utils_keras.dump_image(fusion, img_filename, 'png')

    pass


def save_log_mapping(log_mapping):

    raw_log_file = '%s/%s' % (RESULT_DIR, RAW_LOG_FILENAME_TEMPLATE)
    pickle.dump(log_mapping, open(raw_log_file, 'w'))

    pass


def gtsrb_visualize_label_scan_bottom_right_white_4():

    config.logger.info('loading dataset')
    X_train, Y_train, X_test, Y_test = load_dataset()

    train_generator, test_generator = build_data_loader(
        X_train, Y_train, X_test, Y_test)

    config.logger.info('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)

    # evaluate injected trigger
    # pattern, mask = utils_keras.create_pattern(INPUT_SHAPE, PATTERN_LIST)
    # utils_keras.eval_pattern(
    #     model, X_test, Y_test, pattern, mask,
    #     Y_TARGET, NUM_CLASSES, method=INTENSITY_RANGE, verbose=1)

    # sys.exit(1)

    # initialize visualizer
    visualizer = Visualizer(
        model, intensity_range=INTENSITY_RANGE, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE,
        save_tmp=False, tmp_dir='/mnt/data/bolunwang/backdoor/tmp_gtsrb')

    log_mapping = {}

    # for y_target in [Y_TARGET, 12]:
    y_target_list = range(NUM_CLASSES)
    y_target_list.remove(Y_TARGET)
    y_target_list = [Y_TARGET] + y_target_list
    for y_target in y_target_list:

        config.logger.info('processing label %d' % y_target)

        # X_sample, Y_sample = utils_keras.sample_data(
        #     X_train, Y_train, nb_sample=NB_SAMPLE, num_classes=NUM_CLASSES,
        #     exclude_labels=set([y_target]))

        _, _, logs = visualize_trigger_w_mask(
            visualizer, train_generator, y_target=y_target,
            save_pattern_flag=True)

        log_mapping[y_target] = logs

        save_log_mapping(log_mapping)

    pass


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils_keras.fix_gpu_memory()
    gtsrb_visualize_label_scan_bottom_right_white_4()

    pass


if __name__ == '__main__':

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % HMString(elapsed_time))
