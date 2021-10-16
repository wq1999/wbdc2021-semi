# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
import argparse
import os, sys
import numpy as np
from tqdm import tqdm
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import tensorflow as tf
from tensorflow.python.keras.utils import multi_gpu_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
sys.path.append(os.path.join(BASE_DIR, '../model'))
from conf import *
from evaluation import *
from ple import *
from mmoe import *

dl_path = ROOT_PATH + '/dl'

FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'machine_tag_list_c',
                 'manual_keyword_list_c', 'machine_keyword_list_c', 'manual_tag_list_c']


def parse_opts():
    parser = argparse.ArgumentParser()

    # args for dataloader
    parser.add_argument('--embedding_dim', type=int, default=4)
    parser.add_argument('--test_path', type=str, default='Test Path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_opts()

    embedding_dim = args.embedding_dim
    test_path = args.test_path

    eval_dict = {}

    feed_info_df = pd.read_pickle(dl_path + f'/feed_tag_keyword.pkl')
    # train
    USE_FEAT = ['userid', 'feedid'] + FEA_FEED_LIST[1:]
    train = pd.read_pickle(dl_path + f'/train_data.pkl')[USE_FEAT + ['date_'] + FEA_COLUMN_LIST]

    # test
    test = pd.read_csv(test_path)
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)

    dense_features = ['videoplayseconds']
    sparse_features = [i for i in USE_FEAT if i not in dense_features]

    train[sparse_features] = train[sparse_features].fillna(0)
    train[dense_features] = train[dense_features].fillna(0)

    test[sparse_features] = test[sparse_features].fillna(0)
    test[dense_features] = test[dense_features].fillna(0)

    submit = test[['userid', 'feedid']]

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        train[feat] = lbe.fit_transform(train[feat])
        lbe_mapper = {old_id: new_id for new_id, old_id in enumerate(lbe.classes_)}
        test[feat] = test[feat].apply(lambda x: lbe_mapper.get(x, 0))

    mms = MinMaxScaler(feature_range=(0, 1))
    test[dense_features] = mms.fit_transform(test[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, train[feat].nunique(), embedding_dim=embedding_dim)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    model_input = {name: test[name] for name in feature_names}

    # 4. load model and predict
    model1 = PLE(dnn_feature_columns, num_tasks=7,
                 task_types=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'],
                 task_names=FEA_COLUMN_LIST, num_levels=2, num_experts_specific=8, num_experts_shared=4,
                 expert_dnn_units=[64, 64],
                 gate_dnn_units=[16, 16],
                 tower_dnn_units_lists=[[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]])

    model2 = MMOE(dnn_feature_columns, num_tasks=7,
                  task_types=['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary'],
                  task_names=FEA_COLUMN_LIST, num_experts=32, expert_dnn_units=[64, 64], gate_dnn_units=[32, 32],
                  tower_dnn_units_lists=[[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]])

    save_dir_nn = os.path.join(ROOT_PATH, 'model/nn')
    model1.load_weights(save_dir_nn + '/ple.h5')
    model2.load_weights(save_dir_nn + '/mmoe.h5')

    pred_ans1 = model1.predict(model_input, 2048)
    pred_ans2 = model2.predict(model_input, 2048)

    for i, action in enumerate(FEA_COLUMN_LIST):
        submit[action] = (pred_ans1[i] + pred_ans2[i]) / 2.0

    save_dir = os.path.join(ROOT_PATH, 'submission')
    if not os.path.exists(save_dir):
        print('Create dir: %s' % save_dir)
        os.mkdir(save_dir)

    submit[['userid', 'feedid'] + FEA_COLUMN_LIST].to_csv(save_dir + '/result.csv', index=None, float_format='%.6f')
