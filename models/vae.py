import tensorflow as tf
import numpy as np
import math
import sys
import os
import tf_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
LOSSES_COLLECTION = '_losses'


def placeholder_inputs(batch_size, dimx):
    features = tf.placeholder(tf.float32, shape=(
        batch_size, dimx))
    return features


def get_model(X, is_training, dim_z, bn_decay=None):
    """ Classification model"""
    batch_size = X.get_shape()[0].value
    dimx = img.get_shape()[1].value
    end_points = {}
    encoder_out_l1 = tf_util.fully_connected(
        X, 64, scope='encoder_fc1')
    encoder_out_l2 = tf_util.fully_connected(
        encoder_out_l1, 64, scope='encoder_fc2')
    z = tf_util.fully_connected(encoder_out_l2, dim_z * 2, scope='z')
    z_mu, z_lsgms = tf.split(z, num_or_size_splits=2, axis=1)
    # sample from gaussian distribution
    eps = tf.random_normal(
        tf.stack([tf.shape(X)[0], 16]), 0, 1, dtype=tf.float32)
    z_sample = tf.add(z_mu, tf.multiply(tf.sqrt(tf.exp(dim_z)), eps))
    decoder_out_l1 = tf_util.fully_connected(z_sample, 64, scope='decoder_l1')
    decoder_out_l2 = tf_util.fully_connected(
        decoder_out_l1, 64, scope='decoder_out_l2')
    recon_X = tf_util.fully_connected(
        decoder_out_l2, dimx, scope='recon')
    end_points['z_mu'] = z_mu
    end_points['z_lsgms'] = z_lsgms
    end_points['x'] = X
    end_points['x_recon'] = recon_X

    return recon_X, end_points


def get_loss(end_points, distributions):
    reconstr_loss = 0.5 * \
        tf.reduce_sum(
            tf.pow(tf.subtract(end_points['x_recon'], end_points['x']), 2.0))
    latent_loss = -0.5 * tf.reduce_sum(1 + end_points['z_lsgms']
                                       - tf.square(end_points['z_mu'])
                                       - tf.exp(end_points['z_lsgms']), 1)
    vae_loss = tf.reduce_mean(reconstr_loss + latent_loss)
    return vae_loss
