# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys
import json
from functools import reduce
import pickle
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *

FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'machine_tag_list_c',
                 'manual_keyword_list_c', 'machine_keyword_list_c', 'manual_tag_list_c']

dl_path = os.path.join(ROOT_PATH, 'dl')


def prepare_data():
    dl_path = os.path.join(ROOT_PATH, 'dl')
    feed_info_df = pd.read_pickle(dl_path + f'/feed_tag_keyword.pkl')

    user_action_df = pd.read_csv(USER_ACTION)[["userid", "feedid", 'date_'] + FEA_COLUMN_LIST]

    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')

    # add user-feed embedding
    for f1, f2 in tqdm([['userid', 'feedid']]):
        emb_df = pd.read_pickle(dl_path + '/' + f1 + '_' + f2 + '_embedding.pkl')
        train = train.merge(emb_df, on=f1, how='left')

        del emb_df
        gc.collect()

    if not os.path.exists(dl_path):
        print('Create dir: %s' % dl_path)
        os.mkdir(dl_path)

    train["videoplayseconds"] = np.log(train["videoplayseconds"] + 1.0)

    train.to_pickle(dl_path + f'/train_data.pkl')


if __name__ == "__main__":
    prepare_data()
