#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-22 11:30:01
# @Author  : Shawn Shan (shansixioing@uchicago.edu)
# @Link    : https://www.shawnshan.com/


import keras
import numpy as np


class BackdoorCall(keras.callbacks.Callback):
    def __init__(self, clean_X, clean_Y, adv_gen):
        self.clean_X = clean_X
        self.clean_Y = clean_Y
        self.adv_gen = adv_gen

    def on_epoch_end(self, epoch, logs=None):
        _, clean_acc = self.model.evaluate(self.clean_X, self.clean_Y, verbose=0)
        _, attack_acc = self.model.evaluate_generator(self.adv_gen, steps=100, verbose=0)
        print("Epoch: {} - Clean Acc {:.4f} - Backdoor Success Rate {:.4f}".format(epoch, clean_acc, attack_acc))


def construct_mask_box(target_ls, image_shape, pattern_size=3, margin=1):
    total_ls = {}
    for y_target in target_ls:
        cur_pattern_ls = []
        if image_shape[2] == 1:
            mask, pattern = construct_mask_corner(image_row=image_shape[0],
                                                  image_col=image_shape[1],
                                                  channel_num=image_shape[2],
                                                  pattern_size=pattern_size, margin=margin)
        else:
            mask, pattern = construct_mask_corner(image_row=image_shape[0],
                                                  image_col=image_shape[1],
                                                  channel_num=image_shape[2],
                                                  pattern_size=pattern_size, margin=margin)
        cur_pattern_ls.append([mask, pattern])
        total_ls[y_target] = cur_pattern_ls
    return total_ls


def construct_mask_corner(image_row=32, image_col=32, pattern_size=4, margin=1, channel_num=3):
    mask = np.zeros((image_row, image_col, channel_num))
    pattern = np.zeros((image_row, image_col, channel_num))

    mask[image_row - margin - pattern_size:image_row - margin, image_col - margin - pattern_size:image_col - margin,
    :] = 1
    pattern[image_row - margin - pattern_size:image_row - margin,
    image_col - margin - pattern_size:image_col - margin, :] = [255., 255., 255.]
    return mask, pattern
