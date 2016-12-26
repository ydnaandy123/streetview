from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf

from model import Discriminator

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")

flags.DEFINE_string("mode", "train", "type of mode [test, train, evaluate, complete]")
flags.DEFINE_string("dataset", "indoor", "The name of dataset [cityscapes, inria, indoor]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [16]")
flags.DEFINE_integer("output_size_h", 256, "The size of the output images to produce [192]")
flags.DEFINE_integer("output_size_w", 256, "The size of the output images to produce [512]")

# completion
flags.DEFINE_float("lam", 0.1, "hyper-parameter that controls how import two loss [0.1]")
flags.DEFINE_string("outDir", "completions", "Directory name to save the output [completions]")
flags.DEFINE_string("maskType", "mask", "type of mask [center]")
flags.DEFINE_integer("nIter", 6000, "The number of iteration [1000]")
flags.DEFINE_float("lr", 0.01, "WTF [0.01]")
flags.DEFINE_float("momentum", 0.9, "WTF [0.9]")
# TODO lr and momnetum

# ckt point
flags.DEFINE_string("checkpoint_dir", "/mnt/data/andy/checkpoint/DCGAN_syn",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("dataset_dir", "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples_syn", "Directory name to save the image samples [samples]")

CITYSCAPES_dir = "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom"
CITYSCAPES_mask_dir = "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask"
CITYSCAPES_syn_dir = '/mnt/data/andy/dataset/StreetView_synthesize_ped'
CITYSCAPES_syn_dir_2 = '/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_color'
INRIA_dir = "/mnt/data/andy/dataset/INRIAPerson/96X160H96/Train/pos"
indoor_dir = '/mnt/data/andy/dataset/indoor'

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # Do not take all memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if FLAGS.dataset == 'cityscapes':
            print('Select CITYSCAPES')
            mask_dir = CITYSCAPES_mask_dir
            syn_dir = CITYSCAPES_syn_dir
            FLAGS.output_size_h, FLAGS.output_size_w, FLAGS.is_crop = 192, 512, False
            FLAGS.dataset_dir = CITYSCAPES_dir
        elif FLAGS.dataset == 'inria':
            print('Select INRIAPerson')
            FLAGS.output_size_h, FLAGS.output_size_w, FLAGS.is_crop = 160, 96, False
            FLAGS.dataset_dir = INRIA_dir
        elif FLAGS.dataset == 'indoor':
            print('Select indoor')
            syn_dir = CITYSCAPES_syn_dir
            FLAGS.output_size_h, FLAGS.output_size_w, FLAGS.is_crop = 256, 256, False
            FLAGS.dataset_dir = indoor_dir

        discriminator = Discriminator(sess, batch_size=FLAGS.batch_size, output_size_h=FLAGS.output_size_h, output_size_w=FLAGS.output_size_w, c_dim=FLAGS.c_dim,
                      dataset_name=FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir, dataset_dir=FLAGS.dataset_dir)

        if FLAGS.mode == 'test':
            print('Testing!')
            discriminator.test(FLAGS, syn_dir)
        elif FLAGS.mode == 'train':
            print('Train!')
            discriminator.train(FLAGS, syn_dir)
        elif FLAGS.mode == 'complete':
            print('Complete!')


if __name__ == '__main__':
    tf.app.run()