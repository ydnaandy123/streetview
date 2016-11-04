from __future__ import division
import os
import time
from glob import glob
from six.moves import xrange

from ops import *
from utils import *


class DCGAN(object):
    def __init__(self, sess,
                 batch_size=64, output_size_h=256, output_size_w=512, lam=0.1,
                 z_dim=100, gf_dim=128, df_dim=64, gfc_dim=1024, dfc_dim=1024,
                 y_dim=None, c_dim=3, image_size=256, output_size=256, is_crop=False,
                 dataset_name='default', checkpoint_dir=None):
        # The origin paper gf_dim should be 128
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [128]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
            is_crop: True if images need extra process to crop them.
        """
        self.sess = sess
        self.batch_size = batch_size
        self.sample_size = batch_size  # This variable name need to clarify.
        self.output_size_h = output_size_h
        self.output_size_w = output_size_w

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim # we don't use it.
        self.dfc_dim = dfc_dim # we don't use it.

        self.image_size = image_size # Doesn't matter, the input image is already cropped so the size is equal to out put size.
        self.output_size = output_size  # This not use any more. Since the width and height may not equal.
        self.is_crop = is_crop # False because we already cropped it.
        self.is_grayscale = (c_dim == 1)
        self.c_dim = c_dim

        self.lam = lam

        # batch normalization : deals with poor initialization helps gradient flow
        # one of amazing part of this work
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size_h, self.output_size_w, self.c_dim], name='real_images')
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + [self.output_size_h, self.output_size_w, self.c_dim], name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
            self.G = self.generator(self.z, self.y)
            self.D_logits, self.D = self.discriminator(self.images, self.y, reuse=False)
            self.D_logits_, self.D_ = self.discriminator(self.G, self.y, reuse=True)

            self.sampler = self.sampler(self.z, self.y)
        else:
            # G: generating image
            self.G = self.generator(self.z)
            # D: sigmoid, D_logits: d_h3_lin
            # D: real, D_: fake
            # The logit function is the inverse of the sigmoidal "logistic" function
            self.D, self.D_logits = self.discriminator(self.images)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

            ## wowowoow why can't this move above on D???
            self.sampler = self.sampler(self.z)


        self.z_sum = tf.histogram_summary("z", self.z)
        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)
        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()

        # Gather the variables for each of the models so they can be trained separately.
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

        # Completion.
        self.mask = tf.placeholder(tf.float32, [None] + [self.output_size_h, self.output_size_w, self.c_dim], name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.mul(self.mask, self.G) - tf.mul(self.mask, self.images))), 1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
            data_X, data_y = self.load_mnist()
            sample_images = data_X[0:self.sample_size]
            sample_labels = data_y[0:self.sample_size]
            batch_idxs = min(len(data_X), config.train_size) // config.batch_size

        elif config.dataset == 'cityscapes':
            print ('Select the CITYSCAPES!!!')
            CITYSCAPES_dir = "/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random"
            data = glob(os.path.join(CITYSCAPES_dir, "*.png"))
            batch_idxs = min(len(data), config.train_size) // config.batch_size
            np.random.shuffle(data)  # help or not?

            sample_files = data[0:self.sample_size]
            #sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for sample_file in sample_files]
            sample = [get_image_without_crop(sample_file, is_grayscale=self.is_grayscale) for sample_file in sample_files]
            if (self.is_grayscale):
                sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_images = np.array(sample).astype(np.float32)

        elif config.dataset == 'inria':
            print ('Select the INRIAPerson!!!')
            data_set_dir = "/home/andy/dataset/INRIAPerson/96X160H96/Train/pos"
            data = glob(os.path.join(data_set_dir, "*.png"))
            batch_idxs = min(len(data), config.train_size) // config.batch_size
            np.random.shuffle(data)  # help or not?

            sample_files = data[0:self.sample_size]
            sample = [get_image_without_crop(sample_file, is_grayscale=self.is_grayscale)
                      for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)[:, :, :, 0:3]

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))

        # Gather the variables for each of the models so they can be trained separately.
        # ADAM is often competitive with SGD and (usually)
        # doesn't require hand-tuning of the learning rate, momentum, and other hyper-parameters.
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            np.random.shuffle(data)

            for idx in xrange(0, batch_idxs):
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                if config.dataset == 'mnist':
                    batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
                    errD_real = self.d_loss_real.eval({self.images: batch_images, self.y:batch_labels})
                    errG = self.g_loss.eval({self.z: batch_z, self.y:batch_labels})
                else:
                    batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    #batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for batch_file in batch_files]
                    batch = [get_image_without_crop(batch_file, is_grayscale = self.is_grayscale)
                             for batch_file in batch_files]
                    batch_images = np.array(batch).astype(np.float32)
                    batch_images = batch_images[:, :, :, 0:3]
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #_, summary_str = self.sess.run([g_optim, self.g_sum],
                    #    feed_dict={ self.z: batch_z })
                    #self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z})


                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images, self.y:batch_labels}
                        )
                    else:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images}
                        )
                    save_images(samples, [4, 4],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def test(self, config):
        if config.dataset == 'cityscapes':
            print ('Test the CITYSCAPES!!!')
            CITYSCAPES_dir = "/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random"
            data = glob(os.path.join(CITYSCAPES_dir, "*.png"))
            np.random.shuffle(data)  # help or not?

            sample_files = data[0:self.sample_size]
            sample = [get_image_without_crop(sample_file, is_grayscale=self.is_grayscale) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)

        elif config.dataset == 'inria':
            print ('Select the INRIAPerson!!!')
            data_set_dir = "/home/andy/dataset/INRIAPerson/96X160H96/Train/pos"
            data = glob(os.path.join(data_set_dir, "*.png"))
            np.random.shuffle(data)  # help or not?

            sample_files = data[0:self.sample_size]
            sample = [get_image_without_crop(sample_file, is_grayscale=self.is_grayscale)
                      for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)[:, :, :, 0:3]

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))

        tf.initialize_all_variables().run()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        samples, d_loss, g_loss = self.sess.run(
            [self.sampler, self.d_loss, self.g_loss],
            feed_dict={self.z: sample_z, self.images: sample_images}
        )

        save_images(samples, [4, 4],
                    './test/' + 'yo' + '.png')

    def complete(self, config):
        os.makedirs(os.path.join(config.outDir, 'hats_imgs'), exist_ok=True)
        os.makedirs(os.path.join(config.outDir, 'completed'), exist_ok=True)

        tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
        if config.maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':
            scale = 0.25
            assert(scale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*scale)
            u = int(self.image_size*(1.0-scale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:,:c,:] = 0.0
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        else:
            assert(False)

        for idx in xrange(0, batch_idxs):
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            v = 0

            nRows = np.ceil(batchSz/8)
            nCols = 8
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, batch_mask)
            save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'masked.png'))

            for i in xrange(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: batch_mask,
                    self.images: batch_images,
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                v_prev = np.copy(v)
                v = config.momentum*v - config.lr*g[0]
                zhats += -config.momentum * v_prev + (1+config.momentum)*v
                zhats = np.clip(zhats, -1, 1)

                if i % 50 == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    nRows = np.ceil(batchSz/8)
                    nCols = 8
                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                    completeed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    save_images(completeed[:batchSz,:,:,:], [nRows,nCols], imgName)


    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4
        else:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])            
            h1 = tf.concat(1, [h1, y])
            
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat(1, [h2, y])

            h3 = linear(h2, 1, 'd_h3_lin')
            
            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        if not self.y_dim:
            s, sh, sw = self.output_size, self.output_size_h, self.output_size_w
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            s2_h, s4_h, s8_h, s16_h = int(sh/2), int(sh/4), int(sh/8), int(sh/16)
            s2_w, s4_w, s8_w, s16_w = int(sw/2), int(sw/4), int(sw/8), int(sw/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16_h*s16_w, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16_h, s16_w, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                [self.batch_size, s8_h, s8_w, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                [self.batch_size, s4_h, s4_w, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                [self.batch_size, s2_h, s2_w, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                [self.batch_size, sh, sw, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4) 

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*s4*s4, 'g_h1_lin')))            
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            
            s, sh, sw = self.output_size, self.output_size_h, self.output_size_w
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            s2_h, s4_h, s8_h, s16_h = int(sh/2), int(sh/4), int(sh/8), int(sh/16)
            s2_w, s4_w, s8_w, s16_w = int(sw/2), int(sw/4), int(sw/8), int(sw/16)

            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*s16_h*s16_w, 'g_h0_lin'),
                            [-1, s16_h, s16_w, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s8_h, s8_w, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s4_h, s4_w, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s2_h, s2_w, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, sh, sw, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4)

            # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec
            
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
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
