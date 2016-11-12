#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
import scipy.misc
from glob import glob
import os
from utils import *

'''
flags = tf.app.flags
flags.DEFINE_string("source_dir", "/mnt/data/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_images",
                    "Directory name to read the source images")
flags.DEFINE_string("source_mask_dir", "/mnt/data/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_groundTruth",
                    "Directory name to read the source mask images")
flags.DEFINE_string("target_dir", "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom",
                    "Directory name to read the target images")
flags.DEFINE_string("target_heatmap", "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask",
                    "Directory name to read the images")


flags.DEFINE_integer("sample_size", 1, "The size of sample images [1]")
FLAGS = flags.FLAGS
'''

# TODO: pedestrian mask
#source_dir = '/mnt/data/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_images'
#source_mask_dir = '/mnt/data/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_groundTruth'
# test discrimator
source_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom'
source_mask_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask'
target_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom'
target_heatmap_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask'
output_dir = '/mnt/data/andy/dataset/StreetView_synthesize'
sample_size = 2975

# TODO FLAGS


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO sorting and selection
    # TODO gradient
    data_source = sorted(glob(os.path.join(source_dir, "*.png")))
    source_files = data_source[0:sample_size]
    sources = [get_image(source) for source in source_files]

    # If mask is gray image, the is_gray flag need to be set True
    data_source_mask = sorted(glob(os.path.join(source_mask_dir, "*.png")))
    source_mask_files = data_source_mask[0:sample_size]
    source_masks = [get_image(source_mask, is_gray=False) for source_mask in source_mask_files]

    data_target = (glob(os.path.join(target_dir, "*.png")))
    target_files = data_target[0:sample_size]
    targets = [get_image(target) for target in target_files]

    # TODO heatmap
    data_target_heatmap = sorted(glob(os.path.join(target_heatmap_dir, "*.png")))
    target_heatmap_files = data_target_heatmap[0:sample_size]
    target_heatmaps = [get_image(target_heatmap) for target_heatmap in target_heatmap_files]

    for i in range(sample_size):
        output_dir_cur = os.path.join(output_dir, str(i))
        #output_dir_cur = output_dir
        if not os.path.exists(output_dir_cur):
            os.makedirs(output_dir_cur)
        print(i)
        scipy.misc.imsave(os.path.join(output_dir_cur, 'source{}.png'.format(i)), sources[i])
        scipy.misc.imsave(os.path.join(output_dir_cur, 'source_mask{}.png'.format(i)), source_masks[i])
        scipy.misc.imsave(os.path.join(output_dir_cur, 'target{}.png'.format(i)), targets[i])
        #scipy.misc.imsave(os.path.join(output_dir_cur, 'target_heatmap{}.png'.format(i)), target_heatmaps[i])

        img_ret = blend(targets[i], sources[i], source_masks[i], offset=(0, 0))
        if img_ret is not False:
            scipy.misc.imsave(os.path.join(output_dir, 'img_ret{}.png'.format(i)), img_ret)


if __name__ == '__main__':
    main()
