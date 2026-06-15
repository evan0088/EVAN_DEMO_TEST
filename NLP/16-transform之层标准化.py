import torch
import torch.nn as nn
import math
from input import *

# 注意力机制函数attention 实现思路分析
# q=k=v -> 自注意力
# q!=k=v -> 编码器-解码器一般注意力
"""
缩放点积注意力三步流程:
1. q*k^T/sqrt(d_k) -> 注意力分数
2. 注意力分数经过softmax计算 -> 注意力权重
3. 注意力权重和v进行三维矩阵运算 -> 动态c
"""


# 掩码张量 -> 将注意力分数矩阵相应位置的值替换为-inf

def attention(query, key, value, mask=None, dropout=None):
    """
    注意力计算规则的封装
    :param query: 输入层的输出/解码器自注意力子层输出
    :param key: 输入层的输出/编码器的输出
    :param value: 输入层的输出/编码器的输出
    :param mask: 掩码矩阵
    :param dropout: dropout层对象, 不是置零概率p
    :return:
    """
    # 1 求查询张量特征尺寸大小
    # query -> (句子数, 句子长度, 词维度)
    # -1: 词维度
    d_k = query.size()[-1]
    # print('d_k--->', d_k)

    # 2 求查询张量q的权重分布socres    q@k^T /math.sqrt(d_k)
    # [2,4,512] @ [2,512,4] --->[2,4,4]
    # 注意: key.transpose(-2, -1), 使用-2和-1, 后续多头注意力中 query和key变成[2,8,4,64]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('scores1--->', scores.shape, scores)

    # 3 是否对权重分布scores 进行 masked_fill
    if mask is not None:
        # 根据mask矩阵0的位置 对scores矩阵对应位置进行掩码
        # mask:一个布尔张量（True或False）
        # value:在自然语言处理中，将填充部分的注意力分数设置为一个极小的值（如-1e9），使其在Softmax 后接近0。
        scores = scores.masked_fill(mask=(mask == 0), value=-1e9)
        # print('scores2--->', scores.shape, scores)

    # 4 求查询张量q的权重分布 softmax
    p_attn = torch.softmax(scores, dim=-1)
    # print('p_attn--->', p_attn.shape, p_attn)

    # 5 是否对p_attn进行dropout
    if dropout is not None:
        # dropout(p_attn) -> 对象名(参数值)  调用对象
        p_attn = dropout(p_attn)

    # 返回 查询张量q的注意力结果表示 bmm-matmul运算, 注意力查询张量q的权重分布p_attn
    # [2,4,4]*[2,4,512] --->[2,4,512]
    attn_c = torch.matmul(p_attn, value)
    # print('attn_c--->', attn_c.shape, attn_c)
    return attn_c, p_attn


# 测试
def dm_test_attention():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 0, 0]])
    # 输入部分的Embeddings类
    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)

    dropout_p = 0.1  # 置0概率为0.1
    max_len = 60  # 句子最大长度

    # 输入部分的PositionalEncoding类
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embedded_result)

    # q=k=v -> 输入层的输出
    query = key = value = pe_result  # torch.Size([2, 4, 512])
    # 不带填充掩码
    attn1, p_attn1 = attention(query, key, value)
    print("注意力权重 p_attn2--->", p_attn1.shape, '\n', p_attn1)
    print("注意力表示结果 attn1--->", attn1.shape, '\n', attn1)
    print('=' * 80)

    # print('编码阶段 对注意力权重分布 做掩码')
    # x -> (句子数, 句子长度) -> (2, 4)
    # 了解, 听不懂记住就可以
    # 正确升维 unsqueeze(1): 在维度1上增加一个维度，维度1的值为1 (句子数, 1, 句子长度)  -> 广播机制->(2, 4, 4)
    # 错误升维 unsqueeze(2): 在维度2上增加一个维度，维度2的值为1 (句子数, 句子长度, 1)  -> 广播机制->(2, 4, 4)
    # 错误升维 unsqueeze(0): 在维度0上增加一个维度，维度0的值为1 (1, 句子数, 句子长度)  -> 广播机制->(2, 2, 4)
    mask = (x != 0).type(torch.uint8).unsqueeze(1)
    print('mask--->', mask.shape, mask)
    attn2, p_attn2 = attention(query, key, value, mask=mask)
    print("注意力权重 p_attn2--->", p_attn2.shape, '\n', p_attn2)
    print("注意力表示结果 attn2--->", attn2.shape, '\n', attn2)


import copy


# 多头注意力机制类 MultiHeadedAttention 实现思路分析
# 深度copy模型 输入模型对象和copy的个数 存储到模型列表中
# 深拷贝: 对象之间的内存地址不同, eg:4个线性层的权重参数不同(权重不共享)
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout_p=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 确认数据特征能否被被整除 eg 特征尺寸512 % 头数8
        assert embedding_dim % head == 0, 'head不能被整除'
        # 计算每个头特征尺寸 特征尺寸512 // 头数8 = 64
        # 将512经过线性层映射成64->降维
        self.d_k = embedding_dim // head
        # 多少头数
        self.head = head
        # 克隆四个线性层
        # Q K V 分别线性计算的三个线性层    3层
        # 多个头的注意力表示拼接后线性计算的线性层    1层
        # nn.Linear(embedding_dim, embedding_dim) -> 输入512, 输出512 ->借鉴卷积思想
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # print('self.linears--->', len(self.linears))
        # 注意力权重分布, 多头注意力计算
        self.attn = None
        # dropout层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, query, key, value, mask=None):
        # 求数据多少行 eg:[2,4,512] 则batch_size=2
        # query->(句子数, 句子长度, 词维度)
        batch_size = query.size()[0]
        # print('batch_size--->', batch_size)

        # 数据形状变化[2,4,512] ---> [2,4,8,64] ---> [2,8,4,64]
        # 4代表4个单词 8代表8个头 让句子长度4和句子特征64靠在一起 更有利捕捉句子特征
        # 方式一: 列表推导式
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]
        # print('query--->', query.shape)
        # print('key--->', key.shape)
        # print('value--->', value.shape)

        # 方式二: 普通代码
        # myoutptlist_data = []
        # for model, x in zip(self.linears, (query, key, value)):
        #     print('x--->', x.shape) # [2,4,512]
        #     myoutput = model(x)
        #     print('myoutput--->',    myoutput.shape)    # [2,4,512]
        #     # [2,4,512] --> [2,4,8,64] --> [2,8,4,64]
        #     tmpmyoutput = myoutput.view(batch_size, -1,    self.head, self.d_k).transpose(1, 2)
        #     myoutptlist_data.append( tmpmyoutput )
        # mylen = len(myoutptlist_data)     # mylen:3
        # query = myoutptlist_data[0]         # [2,8,4,64]
        # key = myoutptlist_data[1]             # [2,8,4,64]
        # value = myoutptlist_data[2]         # [2,8,4,64]

        # attention()->4.3章节注意力机制函数
        # 四维矩阵运算-> 前2维不参与计算  词数和词维度参数点乘计算
        # 注意力结果表示x形状 [2,8,4,64] 注意力权重attn形状：[2,8,4,4]
        # attention([2,8,4,64],[2,8,4,64],[2,8,4,64],[1,8,4,4]) ==> x[2,8,4,64], self.attn[2,8,4,4]]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x->动态c, 多头注意力表示结果
        # print('x--->', x.shape, '\n', x)

        # 拼接多头 -> 变回3维形状
        # 数据形状变化 [2,8,4,64] ---> [2,4,8,64] ---> [2,4,512]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        # print('x--->', x.shape, '\n', x)

        # 返回最后变化后的结果 [2,4,512]---> [2,4,512]
        # self.linears[-1]: 获取最后一个线性层对象
        return self.linears[-1](x)


# 测试
# 测试多头注意力机制
def dm_test_MultiHeadedAttention():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维
    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    # 输入部分的Embeddings类
    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)

    dropout_p = 0.1  # 置0概率为0.1
    max_len = 60  # 句子最大长度
    # 输入部分的PositionalEncoding类
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embedded_result)

    head = 8  # 头数head
    query = key = value = pe_result  # torch.Size([2, 4, 512])

    # 输入的掩码张量mask
    # (x != 0).type(torch.uint8): 值为0的位置为0, 不为0的位置为1
    # unsqueeze(1) -> (2,1,4), 后续masked_fill操作时在1轴上进行广播变成(2,4,4)和scores对齐
    # unsqueeze(2) -> (2,1,1,4), 后续masked_fill操作时在1/2轴上进行广播变成(2,8,4,4)和scores对齐
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)

    # 多头注意力机制
    my_mha = MultiHeadedAttention(head, d_model, dropout_p)
    mha_result = my_mha(query, key, value, mask)
    print('多头注意机制后的x', mha_result.shape, '\n', mha_result)
    print('多头注意力机制的注意力权重分布', my_mha.attn.shape)


# 前馈全连接层PositionwiseFeedForward实现思路分析
# 引入更丰富的非线性特征, 让模型的表达能力更强
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        # d_model    第1个线性层输入维度
        # d_ff         第2个线性层输出维度
        super(PositionwiseFeedForward, self).__init__()
        # 定义线性层linear1 linear2 dropout
        # 升维
        self.linear1 = nn.Linear(d_model, d_ff)
        # 降维
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # 数据依次经过第1个线性层 relu激活层 dropout层，然后是第2个线性层
        output1 = self.dropout(torch.relu(self.linear1(x)))
        # print('output1--->', output1.shape)
        output2 = self.linear2(output1)
        # print('output2--->', output2.shape)
        return output2


# 测试
def dm_test_PositionwiseFeedForward():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维
    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    # 输入部分的Embeddings类
    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)

    dropout_p = 0.1  # 置0概率为0.1
    max_len = 60  # 句子最大长度
    # 输入部分的PositionalEncoding类
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embedded_result)

    head = 8  # 头数head
    query = key = value = pe_result  # torch.Size([2, 4, 512])

    # 输入的掩码张量mask
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    my_mha = MultiHeadedAttention(head, d_model, dropout_p)
    mha_result = my_mha(query, key, value, mask)

    # 测试前馈全链接层
    my_pff = PositionwiseFeedForward(d_model=512, d_ff=2048, dropout_p=0.1)
    ff_result = my_pff(mha_result)
    print('x--->', ff_result.shape, ff_result)


# 规范化层或层归一化
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        在词维度对词的每个特征进行标准化
        :param features: 等同于d_model, 词维度
        :param eps: 小常数,防止分母为0
        """
        super(LayerNorm, self).__init__()
        # nn.Parameter():可学习的参数
        # 定义a2 γ规范化层的系数 y=kx+b中的k  初始化1
        self.a2 = nn.Parameter(torch.ones(features))
        # print('self.a2.shape--->', self.a2.shape)
        # 定义b2 β规范化层的系数 y=kx+b中的b  初始化0
        self.b2 = nn.Parameter(torch.zeros(features))
        # print('self.b2.shape--->', self.b2.shape)
        # 小常数
        self.eps = eps

    def forward(self, x):
        # 对数据求均值 保持形状不变
        # -1: 根据最后1个维度计算, 词特征维度
        # (句子数, 词数, 词维度) -> -1:词维度
        # [2,4,512] -> [2,4,1]
        # keepdims: True->保持维度数和原x一致, 3维张量   False->2维
        # print('x--->', x.shape, '\n', x)
        mean = x.mean(dim=-1, keepdims=True)
        # print('mean--->', mean.shape, '\n', mean)
        # 对数据求标准差 保持形状不变
        # [2,4,512] -> [2,4,1]
        std = x.std(-1, keepdims=True)
        # print('std--->', std.shape, '\n', std)
        # 对数据进行标准化变换 反向传播可学习参数a2 b2
        # 注意 * 表示对应位置相乘 不是矩阵运算
        # norm_ret = (x - mean) / (std + self.eps)
        # print('norm_ret--->', norm_ret.shape, '\n', norm_ret)
        x = self.a2 * (x - mean) / (std + self.eps) + self.b2
        return x


# 测试
# 规范化层测试
def dm_test_LayerNorm():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    my_embeddings = Embeddings(vocab, d_model)

    embedded_result = my_embeddings(x)  # [2, 4, 512]


    dropout_p = 0.2  # 置0概率为0.2
    max_len = 60  # 句子最大长度
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embedded_result)

    query = key = value = pe_result  # torch.Size([2, 4, 512])
    # 调用验证

    d_ff = 64
    head = 8

    # 多头注意力机制的输出 作为前馈全连接层的输入
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    my_mha = MultiHeadedAttention(head, d_model, dropout_p)
    mha_result = my_mha(query, key, value, mask)

    my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
    ff_result = my_ff(mha_result)

    # 测试规范化层
    # 特征维度数
    features = d_model = 512
    # 小常数
    eps = 1e-6
    # 实例化规范化层对象
    my_ln = LayerNorm(features, eps)
    # 调用规范化层对象, 对ff_result进行规范化
    ln_result = my_ln(ff_result)
    print('规范化层:', ln_result.shape, ln_result)


# 子层连接结构 子层(前馈全连接层 或者 注意力机制层)+ norm层 + 残差连接
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_p=0.1):
        """
        子层连接
        :param size: 等同于 d_model
        :param dropout_p: 置零概率
        """
        super(SublayerConnection, self).__init__()
        # 参数size 词嵌入维度尺寸大小
        # 参数dropout 置零比率
        self.size = size
        self.dropout_p = dropout_p
        # 定义norm层  对象名=类名(参数值)
        # self.norm = LayerNorm(self.size)
        self.norm = nn.LayerNorm(self.size,eps=1e-6)
        # 定义dropout层
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, sublayer):
        """

        :param x: 当前层输入值
        :param sublayer: 当前层的对象名, 实例化层对象   对象名(x)->调用对象->调用forward方法
        :return: 子层连接的结果
        """
        # 参数x 代表数据
        # sublayer 函数入口地址 子层函数(前馈全连接层 或者 注意力机制层函数的入口地址)
        # 方式1 数据self.norm() -> sublayer() -> self.dropout() + x
        # 通常效果最好
        # myres = x + self.dropout(sublayer(self.norm(x)))
        # 方式2 数据sublayer() -> self.norm() -> self.dropout() + x
        # 不推荐，可能导致训练不稳定
        # myres = x + self.dropout(self.norm(sublayer(x)))
        # 方式3 数据sublayer() -> self.dropout() + x -> self.norm()
        # Transformer的标准实现
        myres = self.norm(x + self.dropout(sublayer(x)))
        return myres


# 测试
def dm_test_SublayerConnection():
    vocab = 1000  # 词表大小是1000
    d_model = 512
    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)  # [2, 4, 512]

    dropout_p = 0.2  # 置0概率为0.2
    max_len = 60  # 句子最大长度
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embedded_result)

    size = 512
    head = 8
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    # 多头自注意力子层
    self_attn = MultiHeadedAttention(head, d_model)
    # 定义匿名函数  函数嵌套调用(后续先调用sublayer, 然后执行内部代码调用self_attn)
    # self_attn(x, x, x): 调用多头自注意力层对象 -> 调用forward方法   x->q k v
    """
    # 方式一:
    def sublayer(x):
        result = self_attn(x, x, x, mask)
        return result
    """
    # 方式二:
    sublayer = lambda x: self_attn(x, x, x, mask)
    print('sublayer--->', sublayer)

    # 子层连接结构
    # 实例化子层连接对象
    my_sc = SublayerConnection(size, dropout_p)
    # 调用子层连接对象
    # 多头自注意力子层输入=输入层的输出=pe_result
    # print('pe_result--->', pe_result)
    sc_result = my_sc(pe_result, sublayer)
    print('sc_result.shape--->', sc_result.shape)
    print('sc_result--->', sc_result)


# 编码器层类 EncoderLayer 实现思路分析
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_p):
        """
        编码器层 = 多头自注意力子层+前馈全连接子层
        :param size: 等同于d_model
        :param self_attn: 多头自注意力层对象
        :param feed_forward: 前馈全连接层对象
        :param dropout_p: 置零概率
        """
        super(EncoderLayer, self).__init__()
        # 实例化多头注意力层对象
        self.self_attn = self_attn
        # 前馈全连接层对象feed_forward
        self.feed_forward = feed_forward
        # size词嵌入维度512
        self.size = size
        self.dropout_p = dropout_p
        # clones两个子层连接结构 self.sublayer = clones(SublayerConnection(size,dropout_p),2)
        # SublayerConnection(self.size, self.dropout_p) -> 实例化子层连接对象
        self.sublayer = clones(SublayerConnection(self.size, self.dropout_p), 2)
        # print('self.sublayer--->', len(self.sublayer), '\n', self.sublayer)

    def forward(self, x, mask):
        """
        前向传播
        :param x: 输入层的输出结果=word_embedding+positional_encoding
        :param mask: 填充掩码
        :return: 编码器层的输出
        """
        # 数据经过第1个子层连接结构 多头自注意力子层
        # 参数x：传入的数据    参数lambda x... : 子函数入口地址
        # self.sublayer -> 列表
        # self.sublayer[0] -> 获取列表中的第1个元素, 就是第1个子层连接对象名
        # print('self.sublayer[0]--->', self.sublayer[0])
        # self.sublayer[0]() -> 对象名(), 调用对象->调用forward方法
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # 数据经过第2个子层连接结构  前馈全连接子层
        # 参数x：传入的数据    self.feed_forward子函数入口地址
        x = self.sublayer[1](x, self.feed_forward)
        return x


# 测试
def dm_test_EncoderLayer():
    vocab = 1000  # 词表大小是1000
    d_model = 512

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    my_embeddings = Embeddings(vocab, d_model)
    embeded_result = my_embeddings(x)  # [2, 4, 512]

    dropout_p = 0.2  # 置0概率为0.2
    max_len = 60  # 句子最大长度
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embeded_result)

    size = 512
    head = 8
    d_ff = 64
    # 实例化多头注意力机制类对象
    self_attn = MultiHeadedAttention(head, d_model)
    # print('self_attn--->', self_attn)
    # 实例化前馈全连接层对象
    my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
    # mask数据
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)

    # 实例化编码器层对象
    my_encoderlayer = EncoderLayer(size, self_attn, my_ff, dropout_p)

    # 数据通过编码层编码
    el_result = my_encoderlayer(pe_result, mask)
    print('el_result.shape', el_result.shape, el_result)


# 编码器类 Encoder 实现思路分析
# init函数 (self, layer, N)
# 实例化多个编码器层对象self.layers     通过方法clones(layer, N)
# 实例化规范化层 self.norm = LayerNorm(layer.size)
# forward函数 (self, x, mask)
# 数据经过N个层 x = layer(x, mask)
#    返回规范化后的数据 return self.norm(x)
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        编码器
        :param layer: 编码器层对象
        :param N: 6, 后续各种大模型调整层数
        """
        # 参数layer 1个编码器层
        # 参数 编码器层的个数
        super(Encoder, self).__init__()
        # 实例化多个编码器层对象
        self.layers = clones(layer, N)
        # print('self.layers--->', len(self.layers), '\n', self.layers)

    def forward(self, x, mask):
        # 遍历self.layers列表, 遍历多个编码器层
        # 数据经过N个层 x = layer(x, mask)
        for layer in self.layers:
            x = layer(x, mask)

        # 返回编码器语义向量
        return x


# 测试
def dm_test_Encoder():
    vocab = 1000  # 词表大小是1000
    d_model = 512

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    my_embeddings = Embeddings(vocab, d_model)
    embeded_result = my_embeddings(x)  # [2, 4, 512]

    dropout_p = 0.2  # 置0概率为0.2
    max_len = 60  # 句子最大长度
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embeded_result)

    size = 512
    head = 8
    d_model = 512
    d_ff = 64

    # 获取位置编码器层 编码以后的结果
    c = copy.deepcopy
    self_attn = MultiHeadedAttention(head, d_model)
    dropout_p = 0.2
    my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
    my_encoderlayer = EncoderLayer(size, c(self_attn), c(my_ff), dropout_p)

    # 编码器中编码器层的个数N
    N = 6
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)

    # 实例化编码器对象
    my_encoder = Encoder(my_encoderlayer, N)
    # 调用编码器对象
    encoder_result = my_encoder(pe_result, mask)
    print('encoder_result.shape--->', encoder_result.shape)
    print('encoder_result--->', encoder_result)
    return encoder_result


if __name__ == '__main__':
    # 注意力机制封装函数
    # dm_test_attention()
    # 多头注意力层
    # dm_test_MultiHeadedAttention()
    # 前馈全连接层
    # dm_test_PositionwiseFeedForward()
    # 规范化层
    # dm_test_LayerNorm()
    # 子层连接结构
    # dm_test_SublayerConnection()
    # 编码器层
    dm_test_EncoderLayer()
    # 编码器
    # dm_test_Encoder()
