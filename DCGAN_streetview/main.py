import os
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")

flags.DEFINE_string("mode", "test", "type of mode [test, train, complete]")
flags.DEFINE_string("dataset", "cityscapes", "The name of dataset [cityscapes, inria]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [16]")
flags.DEFINE_integer("output_size_h", 256, "The size of the output images to produce [256]")
flags.DEFINE_integer("output_size_w", 512, "The size of the output images to produce [512]")

# completion
flags.DEFINE_float("lam", 0.1, "hyper-parameter that controls how import two loss [0.1]")
flags.DEFINE_string("outDir", "completions", "Directory name to save the output [completions]")
flags.DEFINE_string("maskType", "mask", "type of mask [center]")
flags.DEFINE_integer("nIter", 6000, "The number of iteration [1000]")
flags.DEFINE_float("lr", 0.01, "WTF [0.01]")
flags.DEFINE_float("momentum", 0.9, "WTF [0.9]")
# TODO lr and momentum

# almost not use in my application(or be replaced)
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

# change ckt point
flags.DEFINE_string("checkpoint_dir", "/mnt/data/andy/checkpoint/DCGAN",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("dataset_dir", "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

# TODO Fully cityscape dataset and hacker

CITYSCAPES_dir = "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom"
CITYSCAPES_mask_dir = "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_mask"
INRIA_dir = "/mnt/data/andy/dataset/INRIAPerson/96X160H96/Train/pos"


# TODO:  -GAN-improved - VAE - step-by-step GAN
# TODO: Specific generative model?
def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # Do not take all memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # w/ y label
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, y_dim=10, output_size=28,
                          c_dim=1, dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop,
                          checkpoint_dir=FLAGS.checkpoint_dir)
        # w/o y label
        else:
            if FLAGS.dataset == 'cityscapes':
                print 'Select CITYSCAPES'
                mask_dir = CITYSCAPES_mask_dir
                FLAGS.output_size_h, FLAGS.output_size_w, FLAGS.is_crop = 192, 512, False
                FLAGS.dataset_dir = CITYSCAPES_dir
            elif FLAGS.dataset == 'inria':
                print 'Select INRIAPerson'
                FLAGS.output_size_h, FLAGS.output_size_w, FLAGS.is_crop = 160, 96, False
                FLAGS.dataset_dir = INRIA_dir

            dcgan = DCGAN(sess, batch_size=FLAGS.batch_size, output_size_h=FLAGS.output_size_h, output_size_w=FLAGS.output_size_w, c_dim=FLAGS.c_dim,
                          dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop,
                          checkpoint_dir=FLAGS.checkpoint_dir, dataset_dir=FLAGS.dataset_dir)

        if FLAGS.mode == 'test':
            print('Testing!')
            dcgan.test(FLAGS)
        elif FLAGS.mode == 'train':
            print('Train!')
            dcgan.train(FLAGS)
        elif FLAGS.mode == 'complete':
            print('Complete!')
            dcgan.complete(FLAGS, mask_dir)


if __name__ == '__main__':
    tf.app.run()
