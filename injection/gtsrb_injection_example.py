#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-22 11:30:01
# @Author  : Shawn Shan (shansixioing@uchicago.edu)
# @Link    : https://www.shawnshan.com/


import os
import random
import sys

import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

sys.path.append("../")
import utils_backdoor
from injection_utils import *

DATA_DIR = '../data'  # data folder
DATA_FILE = 'gtsrb_dataset.h5'  # dataset file

TARGET_LS = [28]
NUM_LABEL = len(TARGET_LS)
MODEL_FILEPATH = 'gtsrb_backdoor.h5'  # model file
# LOAD_TRAIN_MODEL = 0
NUM_CLASSES = 43
PER_LABEL_RARIO = 0.1
INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
NUMBER_IMAGES_RATIO = 1 / (1 - INJECT_RATIO)
PATTERN_PER_LABEL = 1
INTENSITY_RANGE = "raw"
IMG_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
PATTERN_DICT = construct_mask_box(target_ls=TARGET_LS, image_shape=IMG_SHAPE, pattern_size=4, margin=1)


def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    return X_train, Y_train, X_test, Y_test


def load_traffic_sign_model(base=32, dense=512, num_classes=43):
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(base, (3, 3), padding='same',
                     input_shape=input_shape,
                     activation='relu'))
    model.add(Conv2D(base, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base * 2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 4, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base * 4, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def mask_pattern_func(y_target):
    mask, pattern = random.choice(PATTERN_DICT[y_target])
    mask = np.copy(mask)
    return mask, pattern


def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img


def infect_X(img, tgt):
    mask, pattern = mask_pattern_func(tgt)
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)
    return adv_img, keras.utils.to_categorical(tgt, num_classes=NUM_CLASSES)


class DataGenerator(object):
    def __init__(self, target_ls):
        self.target_ls = target_ls

    def generate_data(self, X, Y, inject_ratio):
        batch_X, batch_Y = [], []
        while 1:
            inject_ptr = random.uniform(0, 1)
            cur_idx = random.randrange(0, len(Y) - 1)
            cur_x = X[cur_idx]
            cur_y = Y[cur_idx]

            if inject_ptr < inject_ratio:
                tgt = random.choice(self.target_ls)
                cur_x, cur_y = infect_X(cur_x, tgt)

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                yield np.array(batch_X), np.array(batch_Y)
                batch_X, batch_Y = [], []


def inject_backdoor():
    train_X, train_Y, test_X, test_Y = load_dataset()  # Load training and testing data
    model = load_traffic_sign_model()  # Build a CNN model

    base_gen = DataGenerator(TARGET_LS)
    test_adv_gen = base_gen.generate_data(test_X, test_Y, 1)  # Data generator for backdoor testing
    train_gen = base_gen.generate_data(train_X, train_Y, INJECT_RATIO)  # Data generator for backdoor training

    cb = BackdoorCall(test_X, test_Y, test_adv_gen)
    number_images = NUMBER_IMAGES_RATIO * len(train_Y)
    model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=10, verbose=0,
                        callbacks=[cb])
    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


if __name__ == '__main__':
    inject_backdoor()
