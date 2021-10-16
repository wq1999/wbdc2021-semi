import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 存储数据的根目录
ROOT_PATH = os.path.join(BASE_DIR, '../data')
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, "wedata/wechat_algo_data1")
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_b.csv")
# 保存模型文件的目录
MODEL_DIR = os.path.join(ROOT_PATH, 'model')
# 保存结果文件的目录
SUMIT_DIR = os.path.join(ROOT_PATH, 'submission')
