#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf

import fcn32_vgg
import utils

import sys

from tensorflow.python.framework import ops

os.environ['CUDA_VISIBLE_DEVICES'] = ''

img1 = skimage.io.imread_collection("./test_data/*.png")
imagesss = np.zeros((5, 1024, 2048, 3))
print imagesss.shape
for i in range(5):
    imagesss[i] = img1[i]
'''
if (self.is_grayscale):
sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
else:
sample_images = np.array(sample).astype(np.float32)
'''
# Do not take all memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #images = tf.placeholder("float")
    #feed_dict = {images: imagesss[0]}
    #batch_images = tf.expand_dims(images, 0)
    batch_images = tf.placeholder("float")
    feed_dict = {batch_images: imagesss}
    
    
    vgg_fcn = fcn32_vgg.FCN32VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])

    scp.misc.imsave('fcn32_downsampled.png', down_color)
    scp.misc.imsave('fcn32_upsampled.png', up_color)
    
    down_color = utils.color_image(down[3])
    up_color = utils.color_image(up[3])

    scp.misc.imsave('fcn32_downsampled5.png', down_color)
    scp.misc.imsave('fcn32_upsampled5.png', up_color)   
   
