import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import pickle
import random
import math
import provider_m2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='vae_m2',
                    help='Model name: lidc or lidc [default: ssl]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=1000,
                    help='Epoch to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Initial learning rate [default: 0.0001]')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=50000,
                    help='Decay step for lr decay [default: 50000]')
parser.add_argument('--decay_rate', type=float, default=0.8,
                    help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
AUG_RATIO = 2 / float(3)

DIMX = 170
DIMZ = 16
ALPHA = 46
NUM_CLASSES = 2

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')





def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            features_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, DIMX)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch'
            # parameter for you every time it trains.
            batch = tf.Variable(0)

            is_labelled = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            X, end_points = MODEL.get_model(
                features_pl, labels_pl, is_training_pl, DIMZ)
            loss = MODEL.get_loss(end_points, is_labelled, ALPHA)
            tf.summary.scalar('loss', loss)

            # correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            # accuracy = tf.reduce_sum(
            #     tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            # tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            # learning_rate = get_learning_rate(batch)

            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                    BASE_LEARNING_RATE, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(BASE_LEARNING_RATE, epsilon=0.01)

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

        ops = {'features': features_pl,
               'labels': labels_pl,
               'is_training': is_training_pl,
               'is_labelled': is_labelled,
               'loss': loss,
               'pred': end_points['y_pred'],
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
    num_batches_l = math.floor(provider_m2.NUM_LABELLED/BATCH_SIZE)
    num_batches_ul = math.floor(provider_m2.NUM_UNLABELLED/BATCH_SIZE)
    num_schedule = [1 for i in range(num_batches_l)]+[0 for i in range(num_batches_ul)]
    random.shuffle(num_schedule)
    list_l = list(range(provider_m2.NUM_LABELLED))
    random.shuffle(list_l)
    list_ul = list(range(provider_m2.NUM_UNLABELLED))
    random.shuffle(list_ul)
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    indx_batch_l = 0
    indx_batch_ul = 0
    for idx in num_schedule:
        if idx == 1:
            start_idx = indx_batch_l * BATCH_SIZE
            end_idx = (indx_batch_l + 1) * BATCH_SIZE
            indx = list_l[start_idx:end_idx]
            indx_batch_l += 1
            batch_data, batch_label = provider_m2.sample_batch_data(True,True,indx)
            is_labelled = True
        else:
            start_idx = indx_batch_ul * BATCH_SIZE
            end_idx = (indx_batch_ul + 1) * BATCH_SIZE
            indx = list_ul[start_idx:end_idx]
            indx_batch_ul += 1
            batch_data, batch_label = provider_m2.sample_batch_data(False, True, indx)
            is_labelled = False

        feed_dict = {ops['features']: batch_data,
                     ops['labels']: batch_label,
                     ops['is_training']: is_training,
                     ops['is_labelled']:is_labelled}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'],
                                                         ops['step'],
                                                         ops['train_op'],
                                                         ops['loss'],
                                                         ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        if idx == 1:
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(len(num_schedule))))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    is_labelled = True
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    num_batches = math.floor(provider_m2.NUM_TEST / BATCH_SIZE)
    list_test = list(range(provider_m2.NUM_TEST))
    random.shuffle(list_test)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        keys_batch = list_test[start_idx:end_idx]
        batch_data, batch_label = provider_m2.sample_batch_data(True,False,keys_batch)
        feed_dict = {ops['features']: batch_data,
                     ops['labels']: batch_label,
                     ops['is_training']: is_training,
                     ops['is_labelled']:is_labelled}
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
            l = int(batch_label[i])
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
    train()
    LOG_FOUT.close()
