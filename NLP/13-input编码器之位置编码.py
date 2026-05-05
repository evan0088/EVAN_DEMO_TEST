# 输入层由词嵌入层+位置编码组成
# x = embedding_x+position_x

# 导入必备的工具包
import torch
# 预定义的网络层torch.nn, 工具开发者已经帮助我们开发好的一些常用层,
# 比如，卷积层, lstm层, embedding层等, 不需要我们再重新造轮子.
import torch.nn as nn
# 数学计算工具包
import math


# todo:1-词嵌入层
class Embeddings(nn.Module):
    # todo:1-1 构造方法, 属性初始化
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        # 参数vocab   词汇表大小
        self.vocab_size = vocab_size
        # 参数d_model 每个词汇的特征尺寸 词嵌入维度  初始词向量或者模型的输出维度 512
        self.d_model = d_model
        # 定义词嵌入层
        # num_embeddings: 词表大小, 可以大于等于词表大小
        # embedding_dim: 词维度
        # padding_idx: 对指定的词下标值不进行词向量转换, 还使用0进行填充
        # 0: 大模型的词表第1个词都是PAD标识符, 表示填充符号, 第1个词下标为0
        self.embed = nn.Embedding(num_embeddings=self.vocab_size,
                                  embedding_dim=self.d_model,
                                  padding_idx=0)

    def forward(self, x):
        # print("x--->", x)
        embedded = self.embed(x)
        print("embedded--->", embedded.shape, embedded)

        # 将x传给self.embed并与根号下self.d_model相乘作为结果返回
        # 1.后续注意力层计算时, 使用缩放点积注意力(除以根号d_model), 当前乘以为了保持模型的方差一致
        # 2.放大embedded结果值, 避免embedded的值和位置编码的量纲差不多, 导致输入层的结果掩盖词向量结果(词信息)
        return embedded * math.sqrt(self.d_model)


# todo:2-位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len=5000):
        # 参数d_model 词嵌入维度 eg: 512个特征
        # 参数max_len 单词token个数 eg: 60个单词
        super(PositionalEncoding, self).__init__()

        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)

        # 思路：位置编码矩阵 + 特征矩阵 相当于给特征增加了位置信息
        # 定义位置编码矩阵PE eg pe[60, 512], 位置编码矩阵和特征矩阵形状是一样的
        # max_len: 句子最大长度, 给句子中每个词添加一个位置信息
        pe = torch.zeros(max_len, d_model)
        # print('pe--->', pe.shape, pe)

        # 定义位置列-矩阵position  数据形状[max_len,1] eg: [0,1,2,3,4...60]^T
        position = torch.arange(0, max_len).unsqueeze(1)
        # print('position--->', position.shape, position)

        # 方式一计算
        _2i = torch.arange(0, d_model, step=2).float()
        pe[:, 0::2] = torch.sin(position / 10000 ** (_2i / d_model))
        pe[:, 1::2] = torch.cos(position / 10000 ** (_2i / d_model))
        # print('pe--->', pe.shape, pe)

        # 形状变化 [60,512]-->[1,60,512]
        # pe位置编码器后续和词嵌入层进行相加, 维度一致, 引入广播变量的机制
        pe = pe.unsqueeze(0)
        # print('pe--->', pe.shape, pe)

        # 把pe位置编码矩阵 注册成模型的持久缓冲区buffer; 模型保存再加载时，可以根模型参数一样，一同被加载
        # 什么是buffer: 对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不参与模型训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        :param x: 词嵌入层的词向量表示
        :return: 词向量+位置编码
        """
        # 注意：输入的x形状2*4*512  pe形状1*60*512  如何进行相加
        # 只需按照x的单词个数 给特征增加位置信息
        print("x--->", x.shape, x)
        # print("x.shape[1]--->", x.shape[1])
        # print("pe--->", self.pe.shape, self.pe)
        print("self.pe[:, :x.shape[1], :]--->", self.pe[:, :x.shape[1], :])
        # x->2个句子  pe->1个  所以使用了广播机制, 让pe变成2个
        # 输入层输出 = 词信息 + 位置信息
        x = x + self.pe[:, :x.shape[1], :]
        print("x--->", x)
        return self.dropout(x)


if __name__ == '__main__':
    # 词表大小
    vocab_size = 1000
    # 词嵌入维度&模型输出维度
    d_model = 512

    # 输入x 句子词下标表示  (句子数, 词下标)
    # 2个句子, 每个句子由4个词组成, 第2个句子进行填充补齐,补0
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 0, 0]])

    # 实例化词嵌入层对象
    embeddings = Embeddings(vocab_size=vocab_size, d_model=d_model)
    # 调用词嵌入层对象
    embedded_x = embeddings(x)
    print("embedded_x--->", embedded_x)

    # 句子最大长度
    max_len = 60
    # 实例化位置编码器对象
    my_pe = PositionalEncoding(d_model=d_model, dropout_p=0.1, max_len=max_len)
    # 调用位置编码器对象, 传入词嵌入层的词向量表示
    pe_result = my_pe(embedded_x)
    print("pe_result--->", pe_result.shape, pe_result)