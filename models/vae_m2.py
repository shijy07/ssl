import tensorflow as tf
import numpy as np
import math
import sys
import os
import tf_util
import tensorflow.contrib.layers as tcl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
LOSSES_COLLECTION = '_losses'

C = - 0.5 * math.log(2 * math.pi)


def KLD(mu, logvar):
    return - 0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))


def bernoulli(p, x):
    epsilon = 1e-8
    return x * tf.log(p + epsilon) + (1 - x) * tf.log(1 - p + epsilon)


def gaussian(x, mu, logvar):
    return C - 0.5 * (logvar + tf.square(x - mu) / tf.exp(logvar))


def std_gaussian(x):
    return C - x ** 2 / 2


def gaussian_std_margin(mu, logvar):
    return C - 0.5 * (tf.square(mu) + tf.exp(logvar))


def gaussian_margin(logvar):
    return C - 0.5 * (1 + logvar)


def placeholder_inputs(batch_size, dimx):
    features_pl = tf.placeholder(tf.float32, shape=(
        batch_size, dimx))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return features_pl, labels_pl


def get_model(X, y, is_training, dim_z, bn_decay=None):
    """ Classification model"""
    batch_size = X.get_shape()[0].value
    dimx = X.get_shape()[1].value
    end_points = {}
    end_points['y_ori'] = y
    y = tcl.one_hot_encoding(y, 2)
    # Build network for q(z|x,y)
    input_xy = tf.concat([X, y], axis=1)
    end_points['y'] = y
    encoder_out_zx_l1 = tf_util.fully_connected(
        input_xy, 64, scope='encoder_zx_fc1', activation_fn=tf.nn.softplus)
    encoder_out_zx_l2 = tf_util.fully_connected(
        encoder_out_zx_l1, 64, scope='encoder_zx_fc2', activation_fn=tf.nn.softplus)
    z = tf_util.fully_connected(encoder_out_zx_l2, dim_z * 2, scope='z')
    z_mu, z_lsgms = tf.split(z, num_or_size_splits=2, axis=1)

    # Build network for q(y|x)
    encoder_out_yx_l1 = tf_util.fully_connected(X, 32, activation_fn=tf.nn.softplus,
                                                scope='encoder_yx_fc1')
    y_pred = tf_util.fully_connected(encoder_out_yx_l1, 2, activation_fn=tf.nn.softplus,
                                     scope='y_pred')
    end_points['y_pred'] = y_pred
    # Sample from p(z), p(y)
    # sample from gaussian distribution for z
    eps = tf.random_normal(
        tf.stack([tf.shape(X)[0], dim_z]), 0, 1, dtype=tf.float32)
    z_sample = tf.add(z_mu, tf.multiply(tf.sqrt(tf.exp(z_lsgms)), eps))


    y_sample = y_pred

    # build q(x|z,y) Bernoulli conditional decoder
    decoder_input = tf.concat([z_sample, y_sample], axis=1, name='concat_zy')
    decoder_out_l1 = tf_util.fully_connected(decoder_input, 64, scope='decoder_l1')
    decoder_out_l2 = tf_util.fully_connected(
        decoder_out_l1, 64, scope='decoder_out_l2')
    recon_X = tf_util.fully_connected(
        decoder_out_l2, dimx,activation_fn=tf.nn.sigmoid, scope='recon')
    recon_X = tf.clip_by_value(recon_X, 1e-8, 1-1e-8)
    end_points['z_mu'] = z_mu
    end_points['z_lsgms'] = z_lsgms
    end_points['x'] = X
    end_points['x_recon'] = recon_X
    return recon_X, end_points



def get_loss(end_points, is_labelled, alpha):
    x = end_points['x']
    x_ = end_points['x_recon']
    reconstr_loss = tf.reduce_mean(tf.reduce_sum(x * tf.log(x_) + (1 - x) * tf.log(1 - x_), 1))
    mu = end_points['z_mu']
    sigma = end_points['z_lsgms']
    KL_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mu)
                                        + tf.square(sigma)
                                        - tf.log(1e-8 + tf.square(sigma)) - 1, 1))
    print(KL_divergence)
    ELBO_ul = KL_divergence-reconstr_loss
    print(ELBO_ul)
    classifier_loss = tf.cond(is_labelled, lambda:tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=end_points['y_pred'], labels=end_points['y_ori'])), lambda:tf.zeros(1))
    vae_loss = tf.cond(is_labelled, lambda:tf.add(ELBO_ul, alpha *classifier_loss), lambda:ELBO_ul)

    with tf.name_scope('summaries'):
        tf.summary.scalar('reconstruct_loss', tf.squeeze(reconstr_loss))
        tf.summary.scalar('KL_divergence', tf.squeeze(KL_divergence))
        tf.summary.scalar('ELBO_ul', tf.squeeze(ELBO_ul))
        tf.summary.scalar('classifier_loss', tf.squeeze(classifier_loss))
    return vae_loss
