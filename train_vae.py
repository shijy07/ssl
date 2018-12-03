import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import pickle
import random
import provider
import tf_util

TRAIN_KEYS_FILE = 'keys_all/train_keys_shuffled.p'
TEST_KEYS_FILE = 'keys_all/testkey.p'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='inception_v1',
                    help='Model name: lidc or lidc [default: lidc]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=10,
                    help='Epoch to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=0.0003,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='rmsprop',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=50000,
                    help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.8,
                    help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
AUG_RATIO = 2 / float(3)

RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

IMG_DIM = 41
NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TEST_KEYS = pickle.load(open(TEST_KEYS_FILE, "rb"))
TRAIN_KEYS = pickle.load(open(TRAIN_KEYS_FILE, "rb"))
#TRAIN_KEYS = provider.getDataKeys(True, False)
#TEST_KEYS = provider.getDataKeys(False,False)
print(len(TRAIN_KEYS))

NUM_TRAIN = len(TRAIN_KEYS)
NUM_TEST = len(TEST_KEYS)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    # CLIP THE LEARNING RATE!
    learing_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train(is_weighted=True):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            img_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, IMG_DIM)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch'
            # parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(
                img_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, is_weighted=is_weighted)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(
                tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            # learning_rate = get_learning_rate(batch)
            learning_rate = 0.04
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.01)
            elif OPTIMIZER == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(
                                learning_rate,
                                RMSPROP_DECAY,
                                momentum=RMSPROP_MOMENTUM,
                                epsilon=RMSPROP_EPSILON)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'img_pl': img_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)

            # Save the variables to disk.
            if (epoch + 1) % 5 == 0:
                eval_one_epoch(sess, ops, test_writer)
                save_path = saver.save(
                    sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    num_batches = NUM_TRAIN / BATCH_SIZE
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    random.shuffle(TRAIN_KEYS)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        keys_batch = TRAIN_KEYS[start_idx:end_idx]
        batch_data, batch_label = provider.loadDataFile(keys_batch)
        feed_dict = {ops['img_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'],
                                                         ops['step'],
                                                         ops['train_op'],
                                                         ops['loss'],
                                                         ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    num_batches = NUM_TEST / BATCH_SIZE
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        keys_batch = TEST_KEYS[start_idx:end_idx]
        batch_data, batch_label = provider.loadDataFile(keys_batch, False)
        feed_dict = {ops['img_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'],
                                                      ops['step'],
                                                      ops['loss'],
                                                      ops['pred']],
                                                     feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1).astype(np.int32)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        for i in range(BATCH_SIZE):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    # log_string('total_seen: %f, %f'%(total_seen_class[0]}
    #                                  total_seen_class[1]))
    log_string('eval avg class acc: %f' % (np.mean(
        np.array(total_correct_class) / np.array(total_seen_class,
                                                 dtype=np.float))))


if __name__ == "__main__":
    train(is_weighted=False)
    LOG_FOUT.close()
