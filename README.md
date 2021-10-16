# **2021中国高校计算机大赛-微信大数据挑战赛复赛代码说明**

本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。 

本次比赛以多个行为预测结果的加权uAUC值进行评分。大赛官方网站：https://algo.weixin.qq.com/

## **1. 环境配置**

- pandas>=1.0.5
- numba>=0.45.1
- scipy>=1.3.1
- deepctr==0.8.5
- tensorflow-gpu==1.13.1
- numpy==1.18.5
- python3
- 其他见requirements.txt文件
## **2. 目录结构**

```python
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/test dataset
|   ├── train, codes for training
|   ├── test, codes for test
|   ├── evaluation.py, (optional) main function for evaluation 
│   ├── model, codes for model architecture
├── data
│   ├── wedata, dataset of the competition
│   ├── submission, prediction result after running inference.sh
│   ├── model, params and model
|   ├── dl, data for nn model
├── config, some file path config
```

## **3. 运行流程**

- 安装环境：sh init.sh
- 进入目录：cd /home/tione/notebook/wbdc2021-semi
- 数据准备和模型训练：sh train.sh
- 预测并生成结果文件：sh inference.sh ../wbdc2021/data/wedata/wechat_algo_data2/test_a.csv

## **4. 模型及特征**

模型：Share_Bottom

- 参数：
  - batch_size: 2048
  - emded_dim: 20
  - num_epochs: 3
  - learning_rate: 0.02
  - bottom_dnn_units=[128, 128, 64], tower_dnn_units_lists=[[64, 32], [64, 32], [64, 32], [64, 32], [64, 32], [64, 32], [64, 32]]
- 特征：
  - sparse特征: userid, feedid, authorid, bgm_singer_id, bgm_song_id, tag, keyword等
  - dense 特征：videoplayseconds和user-feed embedding

模型： MMoE

- 参数：
  - batch_size: 2048
  - emded_dim: 20
  - num_epochs: 2
  - learning_rate: 0.02
  - num_experts=64, expert_dnn_units=[128, 64], gate_dnn_units=[32, 32], tower_dnn_units_lists=[[64, 32], [64, 32], [64, 32], [64, 32], [64, 32], [64, 32], [64, 32]]
- 特征：
  - sparse特征: userid, feedid, authorid, bgm_singer_id, bgm_song_id, tag, keyword等
  - dense 特征：videoplayseconds和user-feed embedding

模型：PLE

- 参数：
  - batch_size: 2048
  - emded_dim: 20
  - num_epochs: 2
  - learning_rate: 0.02
  - num_levels=2, num_experts_specific=8, num_experts_shared=4, expert_dnn_units=[128, 64, 32],
    gate_dnn_units=[16, 16], tower_dnn_units_lists=[[64, 32], [64, 32], [64, 32], [64, 32], [64, 32], [64, 32], [64, 32]]
- 特征：
  - sparse特征: userid, feedid, authorid, bgm_singer_id, bgm_song_id, tag, keyword等
  - dense 特征：videoplayseconds和user-feed embedding

## **5. 算法性能**

- 资源配置：2*P40_48G显存_14核CPU_112G内存
- 预测耗时
  - 总预测时长: 500 s
  - 单个目标行为2000条样本的平均预测时长: 42 ms

## **6. 代码说明**

模型预测部分代码位置如下：

| 路径             | 行数 | 内容                                            |
| :--------------- | :--- | :---------------------------------------------- |
| src/inference.py | 126  | `pred_ans1 = model1.predict(model_input, 2048)` |
| src/inference.py | 127  | `pred_ans2 = model2.predict(model_input, 2048)` |
| src/inference.py | 128  | `pred_ans3 = model3.predict(model_input, 2048)` |

## **7. 相关文献**

- [Multitask learning](http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf)(1998)
- [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)(KDD'18)
- [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)(RecSys'20)

