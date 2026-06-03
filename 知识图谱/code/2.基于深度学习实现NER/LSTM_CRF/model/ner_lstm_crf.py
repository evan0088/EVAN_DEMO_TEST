"""
需求：定义BiLSTM+CRF模型类
思路步骤：
1. 定义模型类：接收embedding_dim, hidden_dim, dropout, word2id, tag2id
    1.1 继承nn.Module，初始化父类
    1.2.基于深度学习实现NER 定义模型的超参数
    1.3 搭建神经网络：①embedding层 ②bi-lstm层 ③隐层转tag线性层 ④crf层

2.基于深度学习实现NER. 实现lstm部分前向传播方法：①进行embedding ②lstm前向传播 ③隐层转tag
    2.基于深度学习实现NER.1 对x进行embedding, 得到x_embedded
    2.基于深度学习实现NER.2.基于深度学习实现NER 对x_embedded进行前向传播
    2.基于深度学习实现NER.3 进行dropout
    2.基于深度学习实现NER.4 基于隐层转tag线性层进行线性变换

3. 实现模型的前向传播方法：接收x, mask。 主要步骤：① ~ ③ + ④维特比解码
    3.1 对x输入lstm进行前向传播
    3.2.基于深度学习实现NER 进行掩码计算，把padding部分发射分数置0
    3.3 计算维特比解码结果得到最优路径

4. 实现模型的损失函数，用于模型训练：：接收x, mask, 主要步骤：① ~ ③ + ④crf前向算法
    4.1对x输入lstm进行前向传播
    4.2.基于深度学习实现NER 进行掩码计算，把padding部分发射分数置0
    4.3 计算损失

"""

# 1. 定义模型类：接收embedding_dim, hidden_dim, dropout, word2id, tag2id
import torch
import torch.nn as nn
from TorchCRF import CRF
import config
import utils.common as common
from utils.data_loader import get_data



# 1.1 继承nn.Module，初始化父类
class NERLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM_CRF, self).__init__()
        # 1.2.基于深度学习实现NER 定义模型的超参数
        self.name = "BiLSTM_CRF"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)

        # 1.3 搭建神经网络：①embedding层 ②bi-lstm层 ③隐层转tag线性层 ④crf层
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # CRF
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        # TODO 因为CRF模型中可训练参数部分是转移分数矩阵，转移分数矩阵的大小是[tag_size, tag_size]
        # 所以这里构建CRF实例的时候，需要传入这个参数
        self.crf = CRF(self.tag_size)

    def forward(self, x, mask):
        # lstm模型得到的结果
        # 3.1 对x输入lstm进行前向传播, 得到发射分数
        emission_score = self.get_lstm2linear(x)
        # 3.2.基于深度学习实现NER 进行掩码计算，把padding部分发射分数置0
        # TODO 这里手动对emission_score基于真实长度做了截断
        emission_score = emission_score * mask.unsqueeze(-1)
        # 3.3 计算维特比解码结果得到最优路径
        # TODO 这里传入的mask需要是BoolTensor， 常见的整型的，这里需要注意
        best_path = self.crf.viterbi_decode(emission_score, mask)
        # TODO best_path返回的是list，并且是二维的：[batch_size, seq_len]
        return best_path

    def log_likelihood(self, x, tags, mask):
        # lstm模型得到的结果
        # 4.1对x输入lstm进行前向传播
        outputs = self.get_lstm2linear(x)
        # 4.2.基于深度学习实现NER 进行掩码计算，把padding部分发射分数置0
        outputs = outputs * mask.unsqueeze(-1)
        # 计算损失
        # 4.3 计算损失
        # TODO: self.crf 得到的是loss函数的负数
        # 方便记忆：self.crf = log( P(Sreal) )
        # 损失函数是= - log( P(Sreal) )
        return - self.crf(outputs, tags, mask)

    def get_lstm2linear(self, x):
        # 2.基于深度学习实现NER.1 对x进行embedding, 得到x_embedded
        embedding = self.word_embeds(x)
        # 2.基于深度学习实现NER.2.基于深度学习实现NER 对x_embedded进行前向传播
        outputs, hidden = self.lstm(embedding)
        # 2.基于深度学习实现NER.3 进行dropout
        outputs = self.dropout(outputs)
        # 2.基于深度学习实现NER.4 基于隐层转tag线性层进行线性变换
        outputs = self.hidden2tag(outputs)
        return outputs

if __name__ == '__main__':
    # 1. 获取word2id
    datas, word2id = common.build_data()
    conf = config.Config()
    # 2.基于深度学习实现NER. 基于dataloader获取编码以后的x,mask,labels
    train_data_loader, dev_data_loader = get_data()
    model = NERLSTM_CRF(embedding_dim=conf.embedding_dim, hidden_dim=conf.hidden_dim, dropout=conf.dropout,
                    word2id=word2id, tag2id=conf.tag2id)

    for input_ids_padded, labels_padded, attention_mask in dev_data_loader:
        outputs = model(input_ids_padded, attention_mask)
        print(f'input_ids_padded: {input_ids_padded.shape}')
        print(f'attention_mask: {attention_mask.shape}')
        print(f'outputs: {outputs}')
        # TODO 对于CRF模型，attention_mask需要是BoolTensor
        attention_mask = attention_mask.to(torch.bool)
        # TODO 返回的结果是每个token对应的loss, 所以这里我们取一个batch的平均值
        loss = model.log_likelihood(input_ids_padded, labels_padded, attention_mask).mean()
        print(f'loss: {loss}')
        print('*' * 100)
