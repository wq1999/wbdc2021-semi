# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
import argparse
import os, sys
import numpy as np
from tqdm import tqdm
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
import tensorflow as tf
import pickle
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
sys.path.append(os.path.join(BASE_DIR, '../model'))
from conf import *
from evaluation import *
from ple import *

dl_path = ROOT_PATH + '/dl'

FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'machine_tag_list_c',
                 'manual_keyword_list_c', 'machine_keyword_list_c', 'manual_tag_list_c']
USER_FEED = ['{}_{}_emb_{}'.format('userid', 'feedid', i) for i in range(16)]


def parse_opts():
    parser = argparse.ArgumentParser()

    # args for dataloader
    parser.add_argument('--embedding_dim', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_opts()

    embedding_dim = args.embedding_dim

    eval_dict = {}

    dl_path = os.path.join(ROOT_PATH, 'dl')
    USE_FEAT = ['userid', 'feedid'] + FEA_FEED_LIST[1:] + USER_FEED

    # read train data
    train = pd.read_pickle(dl_path + f'/train_data.pkl')[USE_FEAT + ['date_'] + FEA_COLUMN_LIST]

    trn = train[train['date_'] < 14]
    val = train[train['date_'] == 14]

    dense_features = ['videoplayseconds'] + USER_FEED
    sparse_features = [i for i in USE_FEAT if i not in dense_features]

    data = pd.concat((trn, val)).reset_index(drop=True)

    data[sparse_features] = data[sparse_features].fillna(0)
    data[dense_features] = data[dense_features].fillna(0)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim) for feat in
                              sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train = data.iloc[:trn.shape[0]].reset_index(drop=True)
    val = data.iloc[trn.shape[0]:].reset_index(drop=True)
    userid_list = val['userid'].astype(str).tolist()

    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}

    train_labels = [train[y].values for y in FEA_COLUMN_LIST]
    val_labels = [val[y].values for y in FEA_COLUMN_LIST]

    # 4. define model, train, evaluate
    model = PLE(dnn_feature_columns, num_tasks=7,
                task_types=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'],
                task_names=FEA_COLUMN_LIST, num_levels=2, num_experts_specific=8, num_experts_shared=4,
                expert_dnn_units=[64, 64],
                gate_dnn_units=[16, 16],
                tower_dnn_units_lists=[[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]])

    # optimizer
    optim = tf.keras.optimizers.Adagrad(lr=0.01 * 2)

    model.compile(optimizer=optim, loss='binary_crossentropy')

    result = []
    best = 0.
    for epoch in range(2):
        history = model.fit(train_model_input, train_labels,
                            batch_size=512 * 4, epochs=1, verbose=1)

        val_pred_ans = model.predict(val_model_input, batch_size=512 * 4)
        weight_auc = evaluate_deepctr(val_labels, val_pred_ans, userid_list, FEA_COLUMN_LIST)

        if weight_auc > best:
            best = weight_auc
            # save model
            save_dir = os.path.join(ROOT_PATH, 'model')
            if not os.path.exists(save_dir):
                print('Create dir: %s' % save_dir)
                os.mkdir(save_dir)

            save_dir_nn = os.path.join(ROOT_PATH, 'model/nn')

            if not os.path.exists(save_dir_nn):
                print('Create dir: %s' % save_dir_nn)
                os.mkdir(save_dir_nn)
            model.save_weights(save_dir_nn + '/ple.h5')

        result.append(weight_auc)

    del train, val, train_model_input, val_model_input
    gc.collect()

    # 5. train on all data
    model_input = {name: data[name] for name in feature_names}

    labels = [data[y].values for y in FEA_COLUMN_LIST]

    model.load_weights(save_dir_nn + '/ple.h5')

    history = model.fit(model_input, labels, batch_size=512 * 4, epochs=1, verbose=1)

    model.save_weights(save_dir_nn + '/ple.h5')

    print('Offline Weight AUC:', max(result))
