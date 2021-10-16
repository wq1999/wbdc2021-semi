import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import gc
import os
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *

pd.set_option('display.max_columns', None)


# user feed embedding
def emb(df, f1, f2):
    emb_size = 16
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, vector_size=emb_size, window=8, min_count=1, sg=1, seed=2021, epochs=10)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)

    df_emb = pd.DataFrame(emb_matrix)
    df_emb.columns = ['{}_{}_emb_{}'.format(
        f1, f2, i) for i in range(emb_size)]

    tmp = pd.concat([tmp, df_emb], axis=1)

    del model, emb_matrix, sentences
    return tmp


# 读取训练集
train = pd.read_csv(USER_ACTION)
df = train

# user-feed embedding
for f1, f2 in tqdm([['userid', 'feedid']]):
    f_path = os.path.join(ROOT_PATH, 'dl')
    emb_df = emb(df, f1, f2)
    emb_df.to_pickle(f_path + '/' + f1 + '_' + f2 + '_embedding.pkl')
