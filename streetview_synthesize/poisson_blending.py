#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.sparse
import scipy.misc
from glob import glob
import os
from utils import *
from random import randint

import numpy as np
import scipy.sparse
import pyamg


def get_image(image_dir, is_gray=False):
    if is_gray:
        return np.array(scipy.misc.imread(image_dir, True))
    else:
        return np.array(scipy.misc.imread(image_dir))


# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask


def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(img_target.shape[0] - offset[0], img_source.shape[0]),
        min(img_target.shape[1] - offset[1], img_source.shape[1]))
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(img_target.shape[0], img_source.shape[0] + offset[0]),
        min(img_target.shape[1], img_source.shape[1] + offset[1]))
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])

    # clip and normalize mask image
    # img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    # img_mask = prepare_mask(img_mask)
    ped_sum = np.sum(img_mask)/255
    # 98304 * 0.9 = 88506
    if ped_sum > 98000:
        print('Too small pedstrian {}/{} pixels'.format(ped_sum, 98304))
        return False
    img_mask[img_mask == 255] = True
    img_mask[img_mask != True] = False

    for i in range(3):
        img_source[:, :, i] = np.multiply(img_source[:, :, i], img_mask)
        img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], i] = \
            np.multiply(img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], i],
                        1-img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]])
    img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], :] += img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], :]
    return img_target
    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y, x]:
                index = x + y * region_size[1]
                A[index, index] = 4
                if index + 1 < np.prod(region_size):
                    A[index, index + 1] = -1
                if index - 1 >= 0:
                    A[index, index - 1] = -1
                if index + region_size[1] < np.prod(region_size):
                    A[index, index + region_size[1]] = -1
                if index - region_size[1] >= 0:
                    A[index, index - region_size[1]] = -1
    A = A.tocsr()

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y, x]:
                    index = x + y * region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A, b, verb=False, tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x

    return img_target

# TODO: pedestrian mask
source_dir = '/mnt/data/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_images'
source_mask_dir = '/mnt/data/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_groundTruth'
# test discrimator
#source_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom'
#source_mask_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask'
target_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom'
target_heatmap_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask'
output_dir = '/mnt/data/andy/dataset/StreetView_synthesize_ped'
sample_size = 1

# TODO FLAGS


def main():

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO sorting and selection
    # TODO gradient
    sample_size = 785
    idx = 0
    data_source = sorted(glob(os.path.join(source_dir, "*.png")))
    source_files = data_source[0:sample_size]
    sources = [get_image(source) for source in source_files]

    # If mask is gray image, the is_gray flag need to be set True
    data_source_mask = sorted(glob(os.path.join(source_mask_dir, "*.png")))
    source_mask_files = data_source_mask[0:sample_size]
    source_masks = [get_image(source_mask, is_gray=True) for source_mask in source_mask_files]

    data_target = (glob(os.path.join(target_dir, "*.png")))
    target_files = data_target[0:sample_size]
    targets = [get_image(target) for target in target_files]

    # TODO heatmap
    data_target_heatmap = sorted(glob(os.path.join(target_heatmap_dir, "*.png")))
    target_heatmap_files = data_target_heatmap[0:sample_size]
    target_heatmaps = [get_image(target_heatmap) for target_heatmap in target_heatmap_files]

    sample_size = min(len(data_source), len(data_target), sample_size)
    for i in range(sample_size):
        output_dir_cur = os.path.join(output_dir, str(i))
        #output_dir_cur = output_dir
        if not os.path.exists(output_dir_cur):
            os.makedirs(output_dir_cur)
        print(i)
        scipy.misc.imsave(os.path.join(output_dir_cur, 'source{}.png'.format(i)), sources[i])
        scipy.misc.imsave(os.path.join(output_dir_cur, 'source_mask{}.png'.format(i)), source_masks[i])
        scipy.misc.imsave(os.path.join(output_dir_cur, 'target{}.png'.format(i)), targets[i])
        scipy.misc.imsave(os.path.join(output_dir_cur, 'target_heatmap{}.png'.format(i)), target_heatmaps[i])

        img_ret = blend(targets[i], sources[i], source_masks[i], offset=(randint(10, 60), randint(100, 350)))
        if img_ret is not False:
            scipy.misc.imsave(os.path.join(output_dir, 'img_ret{}.png'.format(i)), img_ret)


if __name__ == '__main__':
    main()
