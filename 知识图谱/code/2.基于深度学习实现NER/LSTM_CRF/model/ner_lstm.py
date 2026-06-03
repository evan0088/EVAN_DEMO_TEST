import torch
import torch.nn as nn
import config
import utils.common as common
from utils.data_loader import get_data

"""
需求：定义BiLSTM模型类
思路步骤：
1. 定义模型类：接收embedding_dim, hidden_dim, dropout, word2id, tag2id
    1.1 继承nn.Module，初始化父类
    1.2.基于深度学习实现NER 定义模型的超参数
    1.3 搭建神经网络：①embedding层 ②bi-lstm层 ③隐层转tag线性层

2.基于深度学习实现NER. 实现模型的前向传播方法：接收x, mask，主要步骤：①embedding ②lstm前向传播 ③隐层转tag
    2.基于深度学习实现NER.1 词嵌入：对x进行embedding, 得到x_embedded
    2.基于深度学习实现NER.2.基于深度学习实现NER 对x_embedded进行前向传播
    2.基于深度学习实现NER.3 对前向传播的结果进行掩码计算
    2.基于深度学习实现NER.4 进行dropout
    2.基于深度学习实现NER.5 基于隐层转tag线性层进行线性变换
"""


class NERLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        # 1.1 继承nn.Module，初始化父类
        super(NERLSTM, self).__init__()

        # 1.2.基于深度学习实现NER 定义模型的超参数
        self.name = "BiLSTM"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)

        # 1.3 搭建神经网络：①embedding层 ②bi-lstm层 ③隐层转tag线性层
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

    def forward(self, x, mask):
        # x[batch_size, seq_len] , mask[batch_size, seq_len]
        # 2.基于深度学习实现NER.1 词嵌入：对x进行embedding, 得到x_embedded
        # TODO embedding层：每一个输入的token(一个字符)以embedding_dim(300)个维度的词向量进行表示
        # x[batch_size, seq_len] -> embedding[batch_size, seq_len, embedding_dim]
        embedding = self.word_embeds(x)
        # 2.基于深度学习实现NER.2.基于深度学习实现NER 对x_embedded进行前向传播
        # TODO lstm层：接收词向量，提取上下文（高阶语义）
        # embedding[batch_size, seq_len,embedding_dim] -> outputs[batch_size, seq_len, hidden_dim]
        outputs, hidden = self.lstm(embedding)
        # 2.基于深度学习实现NER.3 对前向传播的结果进行掩码计算
        # TODO 掩码层：保留有效位置的输出
        # 输入的embed[0.1, 0.2.基于深度学习实现NER, -1, 0] (lstm) -> outputs [-1,-0.5,-0.3, 1] 这里的第四个位置"1"显然不合理
        # 因为第四个位置在输入的时候这个位置是被补齐的是，原先是空的，但是经过lstm的计算，这里有了特征输出
        # 需要点乘掩码矩阵，保留真实长度的输出结果，比如：
        # outputs[-1,-0.5,-0.3, 1] * mask[1,1,1,0] -> [-1,-0.5,-0.3,0 ]
        # outputs[batch_size, seq_len, hidden_dim] * mask[batch_size, seq_len, 1]
        outputs = outputs * mask.unsqueeze(-1)  # 仅保留有效位置的输出
        # 2.基于深度学习实现NER.4 进行dropout
        # TODO dropout：随机失活，只在训练的时候使用
        outputs = self.dropout(outputs)
        # 2.基于深度学习实现NER.5 基于隐层转tag线性层进行线性变换
        # TODO 输出层：把hidden_dim大小的隐层输出（语义、上下文）转成tag_size个分类数量的概率（logits,没有softmax）
        outputs = self.hidden2tag(outputs)
        return outputs


if __name__ == '__main__':
    # 1. 获取word2id
    datas, word2id = common.build_data()
    conf = config.Config()
    # 2.基于深度学习实现NER. 基于dataloader获取编码以后的x,mask,labels
    train_data_loader, dev_data_loader = get_data()
    model = NERLSTM(embedding_dim=conf.embedding_dim, hidden_dim=conf.hidden_dim, dropout=conf.dropout,
                    word2id=word2id, tag2id=conf.tag2id)

    for input_ids_padded, labels_padded, attention_mask in dev_data_loader:
        outputs = model(input_ids_padded, attention_mask)
        print(f'input_ids_padded: {input_ids_padded.shape}')
        print(f'attention_mask: {attention_mask.shape}')
        print(f'outputs: {outputs.shape}')
        print('*' * 100)
