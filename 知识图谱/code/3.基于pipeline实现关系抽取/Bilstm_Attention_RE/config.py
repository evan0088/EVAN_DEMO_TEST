# coding:utf-8
import torch


class Config(object):
    def __init__(self):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'mps'
        self.device = 'cpu'
        self.train_data_path = "/Users/itheima/Documents/黑马/讲课/就业班/知识图谱/广州3/学生端/02-代码/3.基于pipeline实现关系抽取/Bilstm_Attention_RE/data/train.txt"
        self.test_data_path = "/Users/itheima/Documents/黑马/讲课/就业班/知识图谱/广州3/学生端/02-代码/3.基于pipeline实现关系抽取/Bilstm_Attention_RE/data/test.txt"
        self.rel_data_path = "/Users/itheima/Documents/黑马/讲课/就业班/知识图谱/广州3/学生端/02-代码/3.基于pipeline实现关系抽取/Bilstm_Attention_RE/data/relation2id.txt"
        self.embedding_dim = 128
        # 位置编码的维度
        self.pos_dim = 32
        self.hidden_dim = 200
        self.epochs = 50
        self.batch_size = 32
        self.max_len = 70
        self.learning_rate = 1e-3


if __name__ == '__main__':
    config = Config()
    print(config.train_data_path)
    print(config.test_data_path)