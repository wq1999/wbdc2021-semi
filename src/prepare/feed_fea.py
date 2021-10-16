# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys
import json
from functools import reduce

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *

dl_path = os.path.join(ROOT_PATH, 'dl')
if not os.path.exists(dl_path):
    print('Create dir: %s' % dl_path)
    os.mkdir(dl_path)


def machine_tag_process(x):
    if x == '0':
        return 0
    y = x.split(';')
    y_mat = []
    for i in range(len(y)):
        y_list = [int(y[i].split(' ')[0]), float(y[i].split(' ')[1])]
        y_mat.append(y_list)
    y_mat = np.array(y_mat)
    y_mat_sort = y_mat[np.lexsort(-y_mat.T)]
    tag_max = y_mat_sort[0][0]
    return tag_max


def keyword_tag_process(x):
    if x == '0':
        return 0
    y = int(x.split(';')[0])
    return y


def prepare_data():
    feed_info_df = pd.read_csv(FEED_INFO)

    # feed tag,keyword transform
    feed_info_df[['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']] = feed_info_df[
        ['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']].fillna(0)

    for feat in tqdm(['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']):
        feed_info_df[feat + '_c'] = feed_info_df[feat]
        if feat == 'machine_tag_list':
            feed_info_df[feat + '_c'] = feed_info_df[feat + '_c'].astype('str').apply(machine_tag_process)
        else:
            feed_info_df[feat + '_c'] = feed_info_df[feat + '_c'].astype('str').apply(keyword_tag_process)

    feed_info_df.to_pickle(dl_path + f'/feed_tag_keyword.pkl')


if __name__ == "__main__":
    prepare_data()
