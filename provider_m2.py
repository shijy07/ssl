import os
import sys
import numpy as np
import random
from sklearn import preprocessing

TRAIN_DATA_DIR = 'data/train_data.npy'
TEST_DATA_DIR = 'data/test_data.npy'

train_data = np.load(TRAIN_DATA_DIR).item()
test_data= np.load(TEST_DATA_DIR).item()

X_train_all = train_data['features']
y_train_all = train_data['label']

X_test_all = test_data['features']
y_test_all = test_data['label']
min_max_scaler = preprocessing.MinMaxScaler()

X_train = min_max_scaler.fit_transform(X_train_all)
X_test = min_max_scaler.transform(X_test_all)

UNLABEL = -1.0
indx_train_l = np.where(y_train_all!=UNLABEL)
indx_train_ul = np.where(y_train_all==UNLABEL)
X_train_l = X_train_all[indx_train_l]
y_train_l = y_train_all[indx_train_l]
X_train_ul = X_train_all[indx_train_ul]
y_train_ul = y_train_all[indx_train_ul]

indx_test_l =  np.where(y_test_all!=UNLABEL)
X_test_l = X_test_all[indx_test_l]
y_test_l = y_test_all[indx_test_l]

NUM_LABELLED = y_train_l.shape[0]
NUM_UNLABELLED = y_train_ul.shape[0]

NUM_TEST = y_test_l.shape[0]

def sample_batch_data(is_l,is_train,indx):
    if is_l and is_train:
        return X_train_l[indx], y_train_l[indx]
    elif is_train and (not is_l):
        return X_train_ul[indx], np.zeros(len(indx))
    else:
        return X_test_l[indx], y_test_l[indx]


