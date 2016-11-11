#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
import scipy.misc
from glob import glob
import os

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

#source_dir = '/mnt/data/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_images'
#source_mask_dir = '/mnt/data/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_groundTruth'
source_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom'
source_mask_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask'
target_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom'
target_heatmap_dir = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask'
output_dir = './output'
sample_size = 15


def get_image(image_dir, need_augment=False):
    if need_augment:
        gray = np.array(scipy.misc.imread(image_dir, True))
        return gray #np.dstack((gray, gray, gray))
    else:
        img = np.asarray(PIL.Image.open(image_dir))
        img.flags.writeable = True
        return img


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
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    #img_mask = prepare_mask(img_mask)
    # TODO True False
    img_mask[img_mask == 0] = True
    img_mask[img_mask != True] = False
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

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_source = sorted(glob(os.path.join(source_dir, "*.png")))
    source_files = data_source[0:sample_size]
    sources = [get_image(source) for source in source_files]

    # TODO augment
    data_source_mask = sorted(glob(os.path.join(source_mask_dir, "*.png")))
    source_mask_files = data_source_mask[0:sample_size]
    source_masks = [get_image(source_mask, need_augment=False) for source_mask in source_mask_files]

    # TODO sotring
    # TODO gradient
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

        #source_masks[i] = np.array(scipy.misc.imread('nr_001_img_000000025_c0_re.png')).astype(np.float32)
        #yoo = np.array(scipy.misc.imread('nr_001_img_000000025_c0_re.png'))[:, :, 0:3]
        yoo = source_masks[i]
        #yoo.flags.writeable = True
        #yoo = np.dstack((yoo, yoo, yoo))

        scipy.misc.imsave(os.path.join(output_dir_cur, 'source{}.png'.format(i)), sources[i])
        scipy.misc.imsave(os.path.join(output_dir_cur, 'source_mask{}.png'.format(i)), yoo)
        scipy.misc.imsave(os.path.join(output_dir_cur, 'target{}.png'.format(i)), targets[i])
        scipy.misc.imsave(os.path.join(output_dir_cur, 'target_heatmap{}.png'.format(i)), target_heatmaps[i])

        img_ret = blend(targets[i], sources[i], yoo, offset=(0, 0))
        scipy.misc.imsave(os.path.join(output_dir_cur, 'img_ret{}.png'.format(i)), img_ret)


if __name__ == '__main__':
    main()
