import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf
from glob import glob

import fcn32_vgg
import utils

flags = tf.app.flags
# change ckt point
flags.DEFINE_string("checkpoint_dir", "/mnt/data/andy/checkpoint/DCGAN_syn",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("dataset_dir", "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples_syn", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

def get_image(image_dir, is_gray=False):
    if is_gray:
        return np.array(scipy.misc.imread(image_dir, True))
    else:
        return np.array(scipy.misc.imread(image_dir))

def main(_):

    # Do not take all memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        data = glob(os.path.join(FLAGS.dataset_dir, "*.png"))

        sample_size = min(len(data), 3)
        sample_files = data[0:sample_size]
        sample = [get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)[:, :, :, 0:3]

        batch_images = tf.placeholder("float")
        feed_dict = {batch_images: sample_images}

        vgg_fcn = fcn32_vgg.FCN32VGG()
        with tf.name_scope("content_vgg"):
            vgg_fcn.build(batch_images, debug=True)

        print('Finished building Network.')

        init = tf.initialize_all_variables()
        sess.run(tf.initialize_all_variables())

        print('Running the Network')
        tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
        down, up = sess.run(tensors, feed_dict=feed_dict)

        #down_color = utils.color_image(down[0])
        #up_color = utils.color_image(up[0])

        scp.misc.imsave('fcn32_downsampled.png', down[0])
        scp.misc.imsave('fcn32_upsampled.png', up[0])



if __name__ == '__main__':
    tf.app.run()
