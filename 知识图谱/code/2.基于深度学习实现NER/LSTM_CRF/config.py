"""
需求：把项目常用的变量进行统一管理
    企业中的常规做法，形式可能不一样，可能是放在外部文件中或者shell脚本中，但是目标都是为了方便管理
主要参数：
1. 训练设备
2.基于深度学习实现NER. 数据目录
3. 模型超参数
    3.1 输入层大小
    3.2.基于深度学习实现NER 迭代轮数
    3.3 批次大小
    3.4 隐层大小
    3.5 学习率
    3.6 dropout
"""

import os
import torch
import json


class Config(object):
    def __init__(self):
        # 如果是windows或者linux电脑（使用GPU）
        # M1芯片及其以上的电脑（使用GPU）
        # self.device = 'mps'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        self.train_path = r'D:\workspace\project\PYTHON\AI-Learning\知识图谱\code\2.基于深度学习实现NER\LSTM_CRF\data\train.txt'
        self.vocab_path = r'D:\workspace\project\PYTHON\AI-Learning\知识图谱\code\2.基于深度学习实现NER\LSTM_CRF\data\vocab.txt'
        self.embedding_dim = 300
        self.epochs = 5
        self.batch_size = 8
        # self.batch_size = 64
        self.hidden_dim = 256

        # lstm : e-3
        # bert(只微调输出层) : e-5
        # bert(全量微调)    : e-8
        self.lr = 2e-3  # crf的时候，lr可以小点，比如1e-3
        self.dropout = 0.2
        self.model = 'BiLSTM_CRF'
        # self.model = "BiLSTM_CRF"  # 可以只用"BiLSTM"
        self.tag2id = json.load(open(r'D:\workspace\project\PYTHON\AI-Learning\知识图谱\code\2.基于深度学习实现NER\LSTM_CRF\data\tag2id.json'))


if __name__ == '__main__':
    conf = Config()
    print(conf.train_path)
    print(conf.tag2id)
