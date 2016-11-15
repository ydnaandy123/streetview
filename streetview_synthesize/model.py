from __future__ import division
import os
import time
from glob import glob
from six.moves import xrange

from ops import *
from utils import *


class Discriminator(object):
    def __init__(self, sess,
                 batch_size=64, output_size_h=256, output_size_w=512,
                 df_dim=64, c_dim=3,
                 dataset_name='default', checkpoint_dir=None, dataset_dir=None):
        # TODO The origin paper gf_dim should be 128
        # TODO Can sample_size different from batch_size?
        # http://stackoverflow.com/questions/35289773/cannot-convert-a-partially-converted-tensor-in-tensorflow
        self.sess = sess
        self.batch_size = batch_size
        self.output_size_h = output_size_h
        self.output_size_w = output_size_w
        self.c_dim = c_dim
        self.image_shape = [output_size_h, output_size_w, c_dim]
        self.batch_sqrt = np.ceil(np.sqrt(batch_size))

        self.df_dim = df_dim

        # batch normalization : deals with poor initialization helps gradient flow
        # one of amazing part of this work
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.build_model()

    def build_model(self):

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        self.images_false = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='false_images')

        # D: sigmoid, D_logits: d_h3_lin
        # D: real, D_: fake
        # The logit function is the inverse of the sigmoidal "logistic" function
        self.D, self.D_logits = self.discriminator(self.images)
        self.D_, self.D_logits_ = self.discriminator(self.images_false, reuse=True)


        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()
        # Gather the variables for each of the models so they can be trained separately.
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.saver = tf.train.Saver()

    def train(self, config, syn_dir):
        """Train DCGAN"""
        data = glob(os.path.join(config.dataset_dir, "*.png"))
        data_false = glob(os.path.join(syn_dir, "*.png"))
        batch_idxs = min(len(data), len(data_false), config.train_size) // config.batch_size

        # Gather the variables for each of the models so they can be trained separately.
        # ADAM is often competitive with SGD and (usually)
        # doesn't require hand-tuning of the learning rate, momentum, and other hyper-parameters.
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        tf.initialize_all_variables().run()

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            np.random.shuffle(data)
            for idx in xrange(0, batch_idxs):
                batch_images = get_batch_images(data, idx, config.batch_size, is_grayscale=False)
                batch_images_false = get_batch_images(data_false, idx, config.batch_size, is_grayscale=False)

                # Update D network
                self.sess.run([d_optim],
                              feed_dict={self.images: batch_images, self.images_false: batch_images_false})

                errD_fake = self.d_loss_fake.eval({self.images_false: batch_images_false})
                errD_real = self.d_loss_real.eval({self.images: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_fake_loss: %.8f, d_real_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake, errD_real))
                if np.mod(counter, 10) == 1:
                    store_dir = './samples_syn/'
                    feed_dict = {
                        self.images: batch_images,
                        self.images_false: batch_images_false,
                    }
                    D, D_ = self.sess.run([self.D, self.D_], feed_dict=feed_dict)

                    threshold = 0.5
                    true_ids = np.nonzero((np.array(D) > threshold))
                    false_ids = np.nonzero((np.array(D_) < threshold))
                    batch_images[true_ids, :, :, 2] = 1

                    save_images(batch_images, [self.batch_sqrt, self.batch_sqrt], os.path.join(
                        store_dir,
                        '{}_{}_{:d}_{:d}_{:d}_true.png'.format(counter, config.dataset, config.batch_size, config.output_size_h,
                                                            config.output_size_w)))

                    batch_images_false[false_ids, :, :, 0] = 1
                    save_images(batch_images_false, [self.batch_sqrt, self.batch_sqrt], os.path.join(
                        store_dir,
                        '{}_{}_{:d}_{:d}_{:d}_false.png'.format(counter, config.dataset, config.batch_size, config.output_size_h,
                                                            config.output_size_w)))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def test(self, config, syn_dir):
        store_dir = './test_syn/'
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)

        data = glob(os.path.join(config.dataset_dir, "*.png"))
        data_false = glob(os.path.join(syn_dir, "*.png"))

        data += data_false
        np.random.shuffle(data)

        batch_sqrt = np.ceil(np.sqrt(config.batch_size))

        idx = 20
        batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
        batch = [read_image(batch_file) for batch_file in batch_files]
        batch_images = np.array(batch).astype(np.float32)

        tf.initialize_all_variables().run()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        #errD_fake = self.D_.eval({self.images_false: batch_images})
        feed_dict = {
            self.images: batch_images,
        }
        D = self.sess.run([self.D], feed_dict=feed_dict)
        #print(errD_fake, errD_real)
        print(D)

        save_images(batch_images, [batch_sqrt, batch_sqrt], os.path.join(
            store_dir, '{}_{:d}_{:d}_{:d}_test.png'.format(config.dataset, config.batch_size, config.output_size_h,
                                                      config.output_size_w)))

        threshold = 0.5
        false_ids = np.nonzero((np.array(D) < threshold))
        batch_images[false_ids, :, :, 0] = 1
        test_images = batch_images
        save_images(test_images, [batch_sqrt, batch_sqrt], os.path.join(
            store_dir, '{}_{:d}_{:d}_{:d}_eval.png'.format(config.dataset, config.batch_size, config.output_size_h,
                                                      config.output_size_w)))

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4), h4

    def save(self, checkpoint_dir, step, model_name="DCGAN.model"):
        model_name = model_name
        model_dir = "%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size_h, self.output_size_w)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size_h, self.output_size_w)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
