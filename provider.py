import os
import sys
import numpy as np
import lmdb
import random

TRAIN_DATA_DIR = '/home/shijy07/Coding/LIDC/lidc_train_mv_data'
TEST_DATA_DIR = '/home/shijy07/Coding/LIDC/lidc_test_mv_data'

lmdb_env_train = lmdb.open(TRAIN_DATA_DIR, readonly=True)
lmdb_env_test = lmdb.open(TEST_DATA_DIR, readonly=True)


def getDataKeys(train=True, shuffle=True):
    keys = []
    if train:
        lmdb_env = lmdb.open(TRAIN_DATA_DIR, readonly=True)
    else:
        lmdb_env = lmdb.open(TEST_DATA_DIR, readonly=True)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    for key, _ in lmdb_cursor:
        keys.append(key)
    print len(keys)
    if shuffle:
        random.shuffle(keys)
        return keys
    else:
        return keys


def load_img(keys, train):
    batch_size = len(keys)
    batch_data = np.zeros((batch_size, 41, 41, 3))
    batch_label = np.zeros(batch_size)
    if train:
        with lmdb_env_train.begin() as txn:
            i = 0
            for key in keys:
                lmdb_val = txn.get(key)
                example = np.fromstring(lmdb_val, dtype=np.float32)
                example = np.reshape(example, (3, 41, 41))
                example = np.swapaxes(example, 0, 1)
                example = np.swapaxes(example, 1, 2)
                batch_data[i, :, :, :] = example - 0.3
                if key.endswith('1'):
                    label = 1
                else:
                    label = 0
                batch_label[i] = label
                i += 1
    else:
        with lmdb_env_test.begin() as txn:
            i = 0
            for key in keys:
                lmdb_val = txn.get(key)
                example = np.fromstring(lmdb_val, dtype=np.float32)
                example = np.reshape(example, (3, 41, 41))
                example = np.swapaxes(example, 0, 1)
                example = np.swapaxes(example, 1, 2)
                batch_data[i, :, :, :] = example - 0.3
                if key.endswith('1'):
                    label = 1
                else:
                    label = 0
                batch_label[i] = label
                i += 1
    return (batch_data, np.squeeze(batch_label).astype(np.int32))


def loadDataFile(keys, train=True):
    return load_img(keys, train)
