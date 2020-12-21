import logging

import numpy as np
import tensorflow as tf

from writer import Writer
from utils import general_utils as utils


def id_loss_func(y_gt, y_pred):
    return tf.reduce_mean(tf.keras.losses.MAE(y_gt, y_pred))


class Trainer(object):
    def __init__(self, args, model, data_loader):
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)

        self.model = model
        self.data_loader = data_loader

        # lrs & optimizers
        lr = 5e-5 if self.args.resolution == 256 else 1e-5

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.g_gan_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * lr)
        self.w_d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.4 * lr)

        self.im_d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.4 * lr)

        # Losses
        self.gan_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.pixel_loss_func = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)

        self.id_loss_func = id_loss_func

        if args.pixel_mask_type == 'gaussian':
            sigma = int(80 * (self.args.resolution / 256))
            self.pixel_mask = utils.inverse_gaussian_image(self.args.resolution, sigma)
        else:
            self.pixel_mask = tf.ones([self.args.resolution, self.args.resolution])
            self.pixel_mask = self.pixel_mask / tf.reduce_sum(self.pixel_mask)

        self.pixel_mask = tf.broadcast_to(self.pixel_mask, [self.args.batch_size, *self.pixel_mask.shape])

        self.num_epoch = 0
        self.is_cross_epoch = False

        # Lambdas
        if args.unified:
            self.lambda_gan = 0.5
        else:
            self.lambda_gan = 1

        self.lambda_pixel = 0.02

        self.lambda_id = 1
        self.lambda_attr_id = 1
        self.lambda_landmarks = 0.001
        self.r1_gamma = 10

        # Test
        self.test_not_imporved = 0
        self.max_id_preserve = 0
        self.min_lnd_dist = np.inf

    def train(self):
        while self.num_epoch <= self.args.num_epochs:
            self.logger.info('---------------------------------------')
            self.logger.info(f'Start training epoch: {self.num_epoch}')

            if self.args.cross_frequency and (self.num_epoch % self.args.cross_frequency == 0):
                self.is_cross_epoch = True
                self.logger.info('This epoch is cross-face')
            else:
                self.is_cross_epoch = False
                self.logger.info('This epoch is same-face')

            try:
                if self.num_epoch % self.args.test_frequency == 0:
                    self.test()

                self.train_epoch()

            except Exception as e:
                self.logger.exception(e)
                raise

            if self.test_not_imporved > self.args.not_improved_exit:
                self.logger.info(f'Test has not improved for {self.args.not_improved_exit} epochs. Exiting...')
                break

            self.num_epoch += 1

    def train_epoch(self):
        id_loss = 0
        landmarks_loss = 0
        g_w_gan_loss = 0
        pixel_loss = 0
        w_d_loss = 0
        w_loss = 0

        self.logger.info(f'train in epoch: {self.num_epoch}')
        self.model.train()

        use_w_d = self.args.W_D_loss

        # if use_w_d and use_im_d and not self.args.unified:
        if not self.args.unified:
            if self.num_epoch % 2 == 0:
                # This epoch is not using image_D
                use_im_d = False
                # self.logger.info(f'Not using Image D in epoch: {self.num_epoch}')
            if self.num_epoch % 2 != 0:
                # This epoch is not using W_D
                use_w_d = False
                # self.logger.info(f'Not using W_d in epoch: {self.num_epoch}')

        attr_img, id_img, real_w, real_img, matching_ws = self.data_loader.get_batch(is_cross=self.is_cross_epoch)

        # Forward that does not require grads
        id_embedding = self.model.G.id_encoder(id_img)
        src_landmarks = self.model.G.landmarks(attr_img)
        attr_input = attr_img

        with tf.GradientTape(persistent=True) as g_tape:

            attr_out = self.model.G.attr_encoder(attr_input)
            attr_embedding = attr_out

            self.logger.info(f'attr embedding stats- mean: {tf.reduce_mean(tf.abs(attr_embedding)):.5f},'
                             f' variance: {tf.math.reduce_variance(attr_embedding):.5f}')

            z_tag = tf.concat([id_embedding, attr_embedding], -1)
            w = self.model.G.latent_spaces_mapping(z_tag)
            fake_w = w[:, 0, :]

            self.logger.info(
                f'w stats- mean: {tf.reduce_mean(tf.abs(fake_w)):.5f}, variance: {tf.math.reduce_variance(fake_w):.5f}')

            pred = self.model.G.stylegan_s(w)

            # Move to roughly [0,1]
            pred = (pred + 1) / 2

            if use_w_d:
                with tf.GradientTape() as w_d_tape:
                    fake_w_logit = self.model.W_D(fake_w)
                    g_w_gan_loss = self.generator_gan_loss(fake_w_logit)

                    self.logger.info(f'g W loss is {g_w_gan_loss:.3f}')
                    self.logger.info(f'fake W logit: {tf.squeeze(fake_w_logit)}')

                    with g_tape.stop_recording():
                        real_w_logit = self.model.W_D(real_w)
                        w_d_loss = self.discriminator_loss(fake_w_logit, real_w_logit)
                        w_d_total_loss = w_d_loss

                        if self.args.gp:
                            w_d_gp = self.R1_gp(self.model.W_D, real_w)
                            w_d_total_loss += w_d_gp
                            self.logger.info(f'w_d_gp : {w_d_gp}')

                        self.logger.info(f'W_D loss is {w_d_loss:.3f}')
                        self.logger.info(f'real W logit: {tf.squeeze(real_w_logit)}')

            if self.args.id_loss:
                pred_id_embedding = self.model.G.id_encoder(pred)
                id_loss = self.lambda_id * id_loss_func(pred_id_embedding, tf.stop_gradient(id_embedding))
                self.logger.info(f'id loss is {id_loss:.3f}')

            if self.args.landmarks_loss:
                try:
                    dst_landmarks = self.model.G.landmarks(pred)
                except Exception as e:
                    self.logger.warning(f'Failed finding landmarks on prediction. Dont use landmarks loss. Error:{e}')
                    dst_landmarks = None

                if dst_landmarks is None or src_landmarks is None:
                    landmarks_loss = 0
                else:
                    landmarks_loss = self.lambda_landmarks * \
                                     tf.reduce_mean(tf.keras.losses.MSE(src_landmarks, dst_landmarks))
                    self.logger.info(f'landmarks loss is: {landmarks_loss:.3f}')
                    # if landmarks_loss > 5:
                    #     landmarks_loss = 0
                    #     id_loss = 0

            if not self.is_cross_epoch and self.args.pixel_loss:
                l1_loss = self.pixel_loss_func(attr_img, pred, sample_weight=self.pixel_mask)
                self.logger.info(f'L1 pixel loss is {l1_loss:.3f}')

                if self.args.pixel_loss_type == 'mix':
                    mssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(attr_img, pred, 1.0))
                    self.logger.info(f'mssim loss is {l1_loss:.3f}')
                    pixel_loss = self.lambda_pixel * (0.84 * mssim + 0.16 * l1_loss)
                else:
                    pixel_loss = self.lambda_pixel * l1_loss

                self.logger.info(f'pixel loss is {pixel_loss:.3f}')

            g_gan_loss = g_w_gan_loss

            total_g_not_gan_loss = id_loss \
                                   + landmarks_loss \
                                   + pixel_loss \
                                   + w_loss

            self.logger.info(f'total G (not gan) loss is {total_g_not_gan_loss:.3f}')
            self.logger.info(f'G gan loss is {g_gan_loss:.3f}')

        Writer.add_scalar('loss/landmarks_loss', landmarks_loss, step=self.num_epoch)
        Writer.add_scalar('loss/total_g_not_gan_loss', total_g_not_gan_loss, step=self.num_epoch)

        Writer.add_scalar('loss/id_loss', id_loss, step=self.num_epoch)

        if use_w_d:
            Writer.add_scalar('loss/g_w_gan_loss', g_w_gan_loss, step=self.num_epoch)
            Writer.add_scalar('loss/W_D_loss', w_d_loss, step=self.num_epoch)
            if self.args.gp:
                Writer.add_scalar('loss/w_d_gp', w_d_gp, step=self.num_epoch)

        if not self.is_cross_epoch:
            Writer.add_scalar('loss/pixel_loss', pixel_loss, step=self.num_epoch)
            Writer.add_scalar('loss/w_loss', w_loss, step=self.num_epoch)

        if self.args.debug or \
                (self.num_epoch < 1e3 and self.num_epoch % 1e2 == 0) or \
                (self.num_epoch < 1e4 and self.num_epoch % 1e3 == 0) or \
                (self.num_epoch % 1e4 == 0):
            utils.save_image(pred[0], self.args.images_results.joinpath(f'{self.num_epoch}_prediction_step.png'))
            utils.save_image(id_img[0], self.args.images_results.joinpath(f'{self.num_epoch}_id_step.png'))
            utils.save_image(attr_img[0], self.args.images_results.joinpath(f'{self.num_epoch}_attr_step.png'))

            Writer.add_image('input/id image', tf.expand_dims(id_img[0], 0), step=self.num_epoch)
            Writer.add_image('Prediction', tf.expand_dims(pred[0], 0), step=self.num_epoch)

        if total_g_not_gan_loss != 0:
            g_grads = g_tape.gradient(total_g_not_gan_loss, self.model.G.trainable_variables)

            g_grads_global_norm = tf.linalg.global_norm(g_grads)
            self.logger.info(f'global norm G not gan grad: {g_grads_global_norm}')

            self.g_optimizer.apply_gradients(zip(g_grads, self.model.G.trainable_variables))

        if use_w_d:
            g_gan_grads = g_tape.gradient(g_gan_loss, self.model.G.trainable_variables)

            g_gan_grad_global_norm = tf.linalg.global_norm(g_gan_grads)
            self.logger.info(f'global norm G gan grad: {g_gan_grad_global_norm}')

            self.g_gan_optimizer.apply_gradients(zip(g_gan_grads, self.model.G.trainable_variables))

            w_d_grads = w_d_tape.gradient(w_d_total_loss, self.model.W_D.trainable_variables)

            self.logger.info(f'global W_D gan grad: {tf.linalg.global_norm(w_d_grads)}')
            self.w_d_optimizer.apply_gradients(zip(w_d_grads, self.model.W_D.trainable_variables))

        del g_tape

    # Common

    # Test
    def test(self):
        self.logger.info(f'Testing in epoch: {self.num_epoch}')
        self.model.test()

        similarities = {'id_to_pred': [], 'id_to_attr': [], 'attr_to_pred': []}

        fake_reconstruction = {'MSE': [], 'PSNR': [], 'ID': []}
        real_reconstruction = {'MSE': [], 'PSNR': [], 'ID': []}

        if self.args.test_with_arcface:
            test_similarities = {'id_to_pred': [], 'id_to_attr': [], 'attr_to_pred': []}

        lnd_dist = []

        for i in range(self.args.test_size):
            attr_img, id_img = self.data_loader.get_batch(is_train=False, is_cross=True)

            pred, id_embedding, w, attr_embedding, src_lnds = self.model.G(id_img, attr_img)
            image = tf.clip_by_value(pred, 0, 1)

            pred_id = self.model.G.id_encoder(image)
            attr_id = self.model.G.id_encoder(attr_img)

            similarities['id_to_pred'].extend(tf.keras.losses.cosine_similarity(id_embedding, pred_id).numpy())
            similarities['id_to_attr'].extend(tf.keras.losses.cosine_similarity(id_embedding, attr_id).numpy())
            similarities['attr_to_pred'].extend(tf.keras.losses.cosine_similarity(attr_id, pred_id).numpy())

            if self.args.test_with_arcface:
                try:
                    arc_id_embedding = self.model.G.test_id_encoder(id_img)
                    arc_pred_id = self.model.G.test_id_encoder(image)
                    arc_attr_id = self.model.G.test_id_encoder(attr_img)

                    test_similarities['id_to_attr'].extend(
                        tf.keras.losses.cosine_similarity(arc_id_embedding, arc_attr_id).numpy())
                    test_similarities['id_to_pred'].extend(
                        tf.keras.losses.cosine_similarity(arc_id_embedding, arc_pred_id).numpy())
                    test_similarities['attr_to_pred'].extend(
                        tf.keras.losses.cosine_similarity(arc_attr_id, arc_pred_id).numpy())
                except Exception as e:
                    self.logger.warning(f'Not calculating test similarities for iteration: {i} because: {e}')

            # Landmarks
            dst_lnds = self.model.G.landmarks(image)
            lnd_dist.extend(tf.reduce_mean(tf.keras.losses.MSE(src_lnds, dst_lnds), axis=-1).numpy())

            # Fake Reconstruction
            self.test_reconstruction(id_img, fake_reconstruction, display=(i==0), display_name='id_img')

            if self.args.test_real_attr:
                # Real Reconstruction
                self.test_reconstruction(attr_img, real_reconstruction, display=(i==0), display_name='attr_img')

            if i == 0:
                utils.save_image(image[0], self.args.images_results.joinpath(f'test_prediction_{self.num_epoch}.png'))
                utils.save_image(id_img[0], self.args.images_results.joinpath(f'test_id_{self.num_epoch}.png'))
                utils.save_image(attr_img[0],
                                 self.args.images_results.joinpath(f'test_attr_{self.num_epoch}.png'))

                Writer.add_image('test/prediction', image, step=self.num_epoch)
                Writer.add_image('test input/id image', id_img, step=self.num_epoch)
                Writer.add_image('test input/attr image', attr_img, step=self.num_epoch)

                for j in range(np.minimum(3, src_lnds.shape[0])):
                    src_xy = src_lnds[j]  # GT
                    dst_xy = dst_lnds[j]  # pred

                    attr_marked = utils.mark_landmarks(attr_img[j], src_xy, color=(0, 0, 0))
                    pred_marked = utils.mark_landmarks(pred[j], src_xy, color=(0, 0, 0))
                    pred_marked = utils.mark_landmarks(pred_marked, dst_xy, color=(255, 112, 112))

                    Writer.add_image(f'landmarks/overlay-{j}', pred_marked, step=self.num_epoch)
                    Writer.add_image(f'landmarks/src-{j}', attr_marked, step=self.num_epoch)

        # Similarity
        self.logger.info('Similarities:')
        for k, v in similarities.items():
            self.logger.info(f'{k}: MEAN: {np.mean(v)}, STD: {np.std(v)}')

        mean_lnd_dist = np.mean(lnd_dist)
        self.logger.info(f'Mean landmarks L2: {mean_lnd_dist}')

        id_to_pred = np.mean(similarities['id_to_pred'])
        attr_to_pred = np.mean(similarities['attr_to_pred'])
        mean_disen = attr_to_pred - id_to_pred

        Writer.add_scalar('similarity/score', mean_disen, step=self.num_epoch)
        Writer.add_scalar('similarity/id_to_pred', id_to_pred, step=self.num_epoch)
        Writer.add_scalar('similarity/attr_to_pred', attr_to_pred, step=self.num_epoch)

        if self.args.test_with_arcface:
            arc_id_to_pred = np.mean(test_similarities['id_to_pred'])
            arc_attr_to_pred = np.mean(test_similarities['attr_to_pred'])
            arc_mean_disen = arc_attr_to_pred - arc_id_to_pred

            Writer.add_scalar('arc_similarity/score', arc_mean_disen, step=self.num_epoch)
            Writer.add_scalar('arc_similarity/id_to_pred', arc_id_to_pred, step=self.num_epoch)
            Writer.add_scalar('arc_similarity/attr_to_pred', arc_attr_to_pred, step=self.num_epoch)

        self.logger.info(f'Mean disentanglement score is {mean_disen}')

        Writer.add_scalar('landmarks/L2', np.mean(lnd_dist), step=self.num_epoch)

        # Reconstruction
        if self.args.test_real_attr:
            Writer.add_scalar('reconstruction/real_MSE', np.mean(real_reconstruction['MSE']), step=self.num_epoch)
            Writer.add_scalar('reconstruction/real_PSNR', np.mean(real_reconstruction['PSNR']), step=self.num_epoch)
            Writer.add_scalar('reconstruction/real_ID', np.mean(real_reconstruction['ID']), step=self.num_epoch)

        Writer.add_scalar('reconstruction/fake_MSE', np.mean(fake_reconstruction['MSE']), step=self.num_epoch)
        Writer.add_scalar('reconstruction/fake_PSNR', np.mean(fake_reconstruction['PSNR']), step=self.num_epoch)
        Writer.add_scalar('reconstruction/fake_ID', np.mean(fake_reconstruction['ID']), step=self.num_epoch)

        if mean_lnd_dist < self.min_lnd_dist:
            self.logger.info('Minimum landmarks dist achieved. saving checkpoint')
            self.test_not_imporved = 0
            self.min_lnd_dist = mean_lnd_dist
            self.model.my_save(f'_best_landmarks_epoch_{self.num_epoch}')

        if np.abs(id_to_pred) > self.max_id_preserve:
            self.logger.info(f'Max ID preservation achieved! saving checkpoint')
            self.test_not_imporved = 0
            self.max_id_preserve = np.abs(id_to_pred)
            self.model.my_save(f'_best_id_epoch_{self.num_epoch}')

        else:
            self.test_not_imporved += 1

    def test_reconstruction(self, img, errors_dict, display=False, display_name=None):
        pred, id_embedding, w, attr_embedding, src_lnds = self.model.G(img, img)

        recon_image = tf.clip_by_value(pred, 0, 1)
        recon_pred_id = self.model.G.id_encoder(recon_image)

        mse = tf.reduce_mean((img - recon_image) ** 2, axis=[1, 2, 3]).numpy()
        psnr = tf.image.psnr(img, recon_image, 1).numpy()

        errors_dict['MSE'].extend(mse)
        errors_dict['PSNR'].extend(psnr)
        errors_dict['ID'].extend(tf.keras.losses.cosine_similarity(id_embedding, recon_pred_id).numpy())

        if display:
            Writer.add_image(f'reconstruction/{display_name}', pred, step=self.num_epoch)

    # Helpers

    def generator_gan_loss(self, fake_logit):
        """
        G logistic non saturating loss, to be minimized
        """
        g_gan_loss = self.gan_loss_func(tf.ones_like(fake_logit), fake_logit)
        return self.lambda_gan * g_gan_loss

    def discriminator_loss(self, fake_logit, real_logit):
        """
        D logistic loss, to be minimized
        verified as identical to StyleGAN's loss.D_logistic
        """
        fake_gt = tf.zeros_like(fake_logit)
        real_gt = tf.ones_like(real_logit)

        d_fake_loss = self.gan_loss_func(fake_gt, fake_logit)
        d_real_loss = self.gan_loss_func(real_gt, real_logit)

        d_loss = d_real_loss + d_fake_loss

        return self.lambda_gan * d_loss

    def R1_gp(self, D, x):
        with tf.GradientTape() as t:
            t.watch(x)
            pred = D(x)
            pred_sum = tf.reduce_sum(pred)

        grad = t.gradient(pred_sum, x)

        # Reshape as a vector
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean(norm ** 2)
        gp = 0.5 * self.r1_gamma * gp

        return gp
