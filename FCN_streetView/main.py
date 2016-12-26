import os
import scipy as scp
import scipy.misc

import numpy as np
import tensorflow as tf
from glob import glob
import loss
import fcn32_vgg

flags = tf.app.flags

flags.DEFINE_integer("epoch", 10000, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0000001, "Learning rate of for adam [0.000001]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [16]")

flags.DEFINE_string("dataset_dir", "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("label_dir", "/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_label",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("numClass", 33, "The number of class [33]")

flags.DEFINE_integer("image_size_h", 256, "The size of the  images  [192]")
flags.DEFINE_integer("image_size_w", 512, "The size of the  images  [512]")

flags.DEFINE_string("checkpoint_dir", "/mnt/data/andy/checkpoint/FCN",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples_FCN", "Directory name to save the image samples [samples]")

FLAGS = flags.FLAGS
#TODO: utils


def merge(images, size, is_gray=False):
    h, w = images.shape[1], images.shape[2]
    if is_gray:
        img = np.zeros((int(h * size[0]), int(w * size[1])))
    else:
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx // size[1])
        if is_gray:
            img[j * h:j * h + h, i * w:i * w + w] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def get_image(image_dir, is_gray=False):
    if is_gray:
        return np.array(scipy.misc.imread(image_dir, True))
    else:
        return np.array(scipy.misc.imread(image_dir))


def get_label(image_dir, is_gray=False):
    img = np.array(scipy.misc.imread(image_dir))
    h, w = img.shape
    annotations = np.zeros((h, w, FLAGS.numClass), dtype=np.int32)
    for i in range(FLAGS.numClass):
        annotate = np.zeros((h, w), dtype=np.float32)
        annotate[np.nonzero(img == i)] = 1
        annotations[:, :, i] = annotate
    return np.array(annotations)



def main(_):

    # Do not take all memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)

    with tf.Session() as sess:
        batch_sqrt = np.ceil(np.sqrt(FLAGS.batch_size))
        batch_images = tf.placeholder(tf.float32,
                                      [FLAGS.batch_size, FLAGS.image_size_h, FLAGS.image_size_w, 3])
        batch_images_label = tf.placeholder(tf.float32,
                                            [FLAGS.batch_size, FLAGS.image_size_h, FLAGS.image_size_w, FLAGS.numClass])

        vgg_fcn = fcn32_vgg.FCN32VGG()
        with tf.name_scope("content_vgg"):
            #vgg_fcn.build(batch_images, debug=True)
            vgg_fcn.build(rgb=batch_images, train=True, num_classes=FLAGS.numClass,
                          random_init_fc8=True, debug=False)

        fcn_loss = loss.loss(vgg_fcn.upscore, batch_images_label, FLAGS.numClass)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(fcn_loss, global_step=global_step)

        print('Finished building Network.')
        sess.run(tf.initialize_all_variables())

        print('Running the Network')
        #data = sorted(glob(os.path.join(FLAGS.dataset_dir, "*.png")))
        #label = sorted(glob(os.path.join(FLAGS.label_dir, "*.png")))

        data = ['rgb.png']
        label = ['label.png']

        sample_size = min(len(data), FLAGS.batch_size)
        sample_files = data[0:sample_size]
        sample = [get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)[:, :, :, 0:3]

        sample_files_label = label[0:sample_size]
        sample_label = [get_label(sample_file_label) for sample_file_label in sample_files_label]
        sample_images_label = np.array(sample_label).astype(np.float32)

        feed_dict = {batch_images: sample_images, batch_images_label: sample_images_label}
        #tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
        #down, up = sess.run(tensors, feed_dict=feed_dict)

        for step in xrange(FLAGS.epoch):
            _, loss_value = sess.run([train_op, fcn_loss],
                                     feed_dict=feed_dict)
            print("Step: [%5d] loss: %.4f" % (step, loss_value))

            if step % 100 == 0:
                tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
                down, up = sess.run(tensors, feed_dict=feed_dict)
                scp.misc.imsave('results/%3d_fcn32_downsampled.png' % (step/100), down[0])
                scp.misc.imsave('results/%3d_fcn32_upsampled.png' % (step/100), up[0])
                scp.misc.imsave('results/%3d_fcn32_origin.png' % (step/100), merge(sample_images, (batch_sqrt, batch_sqrt), is_gray=False))

        #down_color = utils.color_image(down[0])
        #up_color = utils.color_image(up[0])

        #scp.misc.imsave('fcn32_downsampled.png', utils.color_image(down[0]))
        #scp.misc.imsave('fcn32_upsampled.png', utils.color_image(up[0]))
        #scp.misc.imsave('fcn32_origin.png', utils.color_image(sample_images[0]))



if __name__ == '__main__':
    tf.app.run()
