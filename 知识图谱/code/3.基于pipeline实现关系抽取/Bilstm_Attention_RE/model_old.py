"""
需求：构建一个基于双向长短期记忆网络（BiLSTM）和注意力机制（Attn）的模型，用于处理文本相关任务（如关系分类等）。
思路步骤：
1. 初始化参数：
    1.1 接收词汇表大小、词性维度、标签大小等参数。
    1.2 保存配置中的嵌入维度、位置编码维度、隐藏层维度、批次大小等参数。
2. 定义网络层：
    2.1 定义文本嵌入层、位置编码层1、位置编码层2。
    2.2 定义双向LSTM层，其输入大小为嵌入维度与两倍位置编码维度之和，隐藏层大小为总隐藏层维度的一半。
    2.3 定义线性层，将隐藏层输出映射到标签维度。
    2.4 定义多个Dropout层用于防止过拟合。
    2.5 定义注意力权重参数或线性层用于注意力计算。
3. 实现注意力机制：
    3.1 对输入的隐藏层表示进行双曲正切激活。
    3.2 通过注意力权重计算注意力分数并进行softmax归一化。
    3.3 使用注意力分数对隐藏层表示进行加权求和得到注意力输出。
4. 前向传播：
    4.1 将输入的文本、位置1、位置2分别进行嵌入。
    4.2 拼接嵌入结果并通过Dropout层。
    4.3 将拼接结果输入双向LSTM层得到输出，并再次通过Dropout层。
    4.4 对LSTM输出应用注意力机制，经过激活函数和Dropout后，去除最后一维的冗余维度。
    4.5 通过线性层得到最终输出。
"""

# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_loader import get_loader_data, word2id
from config import Config

# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_ATT(nn.Module):
    def __init__(self, conf, vocab_size, pos_size, tag_size):
        super(BiLSTM_ATT, self).__init__()
        self.batch = conf.batch_size
        self.device = conf.device
        self.vocab_size = vocab_size
        self.embedding_dim = conf.embedding_dim

        self.hidden_dim = conf.hidden_dim

        self.pos_size = pos_size
        self.pos_dim = conf.pos_dim

        self.tag_size = tag_size

        self.word_embeds = nn.Embedding(self.vocab_size,
                                        self.embedding_dim)

        self.pos1_embeds = nn.Embedding(self.pos_size,
                                        self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size,
                                        self.pos_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.pos_dim * 2,
                            hidden_size=self.hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True)

        self.linear = nn.Linear(self.hidden_dim,
                                self.tag_size)

        self.dropout_emb = nn.Dropout(p=0.2)
        self.dropout_lstm = nn.Dropout(p=0.2)
        self.dropout_att = nn.Dropout(p=0.2)

        self.att_weight = nn.Parameter(torch.randn(self.batch,
                                                   1,
                                                   self.hidden_dim).to(self.device))


    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device),
                torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device))

    def attention(self, H):
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight, M), dim=-1)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)

    def forward(self, sentence, pos1, pos2):

        init_hidden = self.init_hidden_lstm()

        embeds = torch.cat((self.word_embeds(sentence),
                            self.pos1_embeds(pos1),
                            self.pos2_embeds(pos2)), 2)

        embeds = self.dropout_emb(embeds)
        embeds = torch.transpose(embeds, 0, 1)

        lstm_out, lstm_hidden = self.lstm(embeds, init_hidden)

        lstm_out = lstm_out.permute(1, 2, 0)

        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out))
        att_out = self.dropout_att(att_out).squeeze()

        result = self.linear(att_out)

        return result
