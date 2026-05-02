# 用于正则表达式
import re

# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# torch中预定义的优化方法工具包
import torch.optim as optim
import time

# 用于随机生成数据
import random
import numpy as np
import matplotlib.pyplot as plt

# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 起始标志 SOS->Start Of Sequence
SOS_token = 0
# 结束标志 EOS->End Of Sequence
EOS_token = 1
# 最大句子长度不能超过10个(包含标点)，用于设置每个句子样本的中间语义张量c长度都为10。
MAX_LENGTH = 10
# 数据文件路径
data_path = "./data/eng-fra-v2.txt"


# todo: 1-文本清洗工具函数
def normalizeString(s: str):
    """字符串规范化函数, 参数s代表传入的字符串"""
    s = s.lower().strip()
    # print('s1--->', s)
    # 在.!?前加一个空格, 即用 “空格 + 原标点” 替换原标点。
    # \1 代表 捕获的标点符号，即 ., !, ? 之一。
    s = re.sub(r"([.!?])", r" \1", s)
    # print('s2--->', s)
    # 用一个空格替换原标点，意味着 标点符号被完全去掉，只留下空格。
    # s = re.sub(r"([.!?])", r" ", s)
    # 使用正则表达式将字符串中 不是 至少1个小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-z.!?]+", r" ", s)
    # print('s3--->', s)
    return s


# todo: 2-加载数据集到内存中, 以及构建英文词表和法文词表
def my_getdata(data_path):
    # 1 按行读文件 open().read().strip().split(\n)
    with open(data_path, "r", encoding="utf-8") as f:
        my_lines = f.read().strip().split("\n")
    # print("my_lines--->", len(my_lines))
    # print("my_lines[:5]--->", my_lines[:5])

    # 2 按行清洗文本 构建语言对 my_pairs
    # 格式 [['英文句子', '法文句子'], ['英文句子', '法文句子'], ['英文句子', '法文句子'], ... ]
    tmp_pair, my_pairs = [], []
    for l in my_lines:
        # print('l.split("\t")--->', l.split("\t"))
        for s in l.split("\t"):
            # 遍历英文和法文句子调用文本清洗工具函数进行清洗
            tmp_pair.append(normalizeString(s))
        my_pairs.append(tmp_pair)
        # 清空tmp_pair, 存储下一个句子的英语和法语的句子对
        tmp_pair = []
    # 方式二: 列表推导式
    # my_pairs = [[normalizeString(s) for s in l.split('\t')] for l in my_lines]
    # print('my_pairs--->', my_pairs)
    print("len(my_pairs)--->", len(my_pairs))
    # 打印前4条数据
    print(my_pairs[:4])
    # 打印第8000条的英文 法文数据
    print("my_pairs[8000][0]--->", my_pairs[8000][0])
    print("my_pairs[8000][1]--->", my_pairs[8000][1])

    # 3 遍历语言对 构建英语单词字典 法语单词字典
    # 3-1 english_word2index english_word_n french_word2index french_word_n
    # SOS->Start Of Sequence, 开始符号
    # EOS->End Of Sequence, 结束符号
    # 真实大模型词表中, 第一个词是PAD标识符, 填充符号
    english_word2index = {"SOS": 0, "EOS": 1}
    # 第三个单词的下标值从2开始
    english_word_n = 2

    french_word2index = {"SOS": 0, "EOS": 1}
    french_word_n = 2

    # 遍历语言对 获取英语单词字典 法语单词字典
    # {单词1:下标1, 单词2:下标2, ...}
    for pair in my_pairs:
        # print("pair--->", pair)
        # print("pair[0].split(' ')--->", pair[0].split(" "))
        for word in pair[0].split(" "):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                # 更新下一个单词的下标值
                english_word_n += 1

        for word in pair[1].split(" "):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1
    # print("english_word2index--->", english_word2index)
    # print("french_word2index--->", french_word2index)

    # 3-2 english_index2word french_index2word
    # # {下标1:单词1, 下标2:单词2, ...}
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}
    # print("english_index2word--->", english_index2word)
    # print("french_index2word--->", french_index2word)
    print("len(english_word2index)-->", len(english_word2index))
    print("len(french_word2index)-->", len(french_word2index))
    print("english_word_n--->", english_word_n, "french_word_n-->", french_word_n)

    return (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    )


# todo: 3-构建张量数据集对象
class MyPairsDataset(Dataset):
    def __init__(self, my_pairs, english_word2index, french_word2index):
        # 样本x
        self.my_pairs = my_pairs
        self.english_word2index = english_word2index
        self.french_word2index = french_word2index
        # 样本条目数
        self.sample_len = len(my_pairs)

    # 获取样本条数
    def __len__(self):
        return self.sample_len

    # 获取第几条 样本数据
    def __getitem__(self, index):
        # print('self.my_pairs--->', my_pairs[:3])
        # 对index异常值进行修正 [0, self.sample_len-1]
        index = min(max(index, 0), self.sample_len - 1)
        # print('index--->', index)

        # 按索引获取 数据样本 x y
        # print('self.my_pairs[index]--->', self.my_pairs[index])
        x = self.my_pairs[index][0]  # 英文句子
        y = self.my_pairs[index][1]  # 法文句子
        # print('x--->', x)
        # print('y--->', y)

        # 样本x 文本数值化
        x = [self.english_word2index[word] for word in x.split(" ")]
        x.append(EOS_token)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)
        # print('tensor_x.shape===>', tensor_x.shape, tensor_x)

        # 样本y 文本数值化
        y = [self.french_word2index[word] for word in y.split(" ")]
        y.append(EOS_token)
        # print('y--->', y)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)
        # 注意 tensor_x tensor_y都是一维数组，通过DataLoader拿出的数据是二维数据
        # print('tensor_y.shape===>', tensor_y.shape, tensor_y)

        # 返回结果
        return tensor_x, tensor_y


# todo: 4-构建编码器类
# class 类名(nn.Module): -> 构建人工神经网络语法
class EncoderRNN(nn.Module):
    """
    对输入序列x英文句子进行编码 -> 捕获英文句子的语义信息
    """

    # todo: 4-1 构造方法, 搭建神经网络模型结构
    def __init__(self, input_size, word_dim, hidden_size):
        super(EncoderRNN, self).__init__()
        # input_size 编码器 词嵌入层单词数 eg：2803
        self.input_size = input_size
        # word_dim 词嵌入维度/初始词向量维度 eg: 512
        self.word_dim = word_dim
        # hidden_size gru层的隐藏层特征数 eg: 256
        self.hidden_size = hidden_size

        # 实例化nn.Embedding层
        # num_embeddings: 词表大小
        # embedding_dim: 词嵌入维度/初始词向量维度
        self.embedding = nn.Embedding(
            num_embeddings=self.input_size, embedding_dim=self.word_dim
        )

        # 实例化nn.GRU层 注意参数batch_first=True->(batch_size, seq_len, hidden_size)
        # input_size: 上一层的输出维度/embedding层的词向量维度
        # hidden_size: 隐藏层维度/gru层的特征维度
        self.gru = nn.GRU(
            input_size=self.word_dim, hidden_size=self.hidden_size, batch_first=True
        )

    # todo: 4-2 前向传播方法, 计算句子语义表示
    def forward(self, input, h0):
        """
        计算句子语义表示
        :param input: 每批的句子数据
        :param h0: 初始的h0
        :return: 所有时间步的隐藏状态值, 最后一个时间步的隐藏状态值
        """
        # print("input--->", input.shape, input)
        # 数据经过词嵌入层 数据形状 [1,6] --> [1,6,256]
        # 返回的形状一定是三维数据集 batch_first=True->(句子数, 句子长度, 词维度)
        embedded = self.embedding(input)
        # print("embedded--->", embedded.shape, embedded)

        # 数据经过gru层 数据形状 gru([1,6,256],[1,1,256]) --> [1,6,256] [1,1,256]
        # print("h0--->", h0.shape, h0)
        output, hn = self.gru(embedded, h0)
        # print("output--->", output.shape, output)
        # print("hn--->", hn.shape, hn)
        return output, hn

    # todo: 4-3 线性初始化h0, 也可以隐性初始化h0,就不需要inithiedden方法了
    def inithidden(self):
        # 将隐藏层张量初始化成为1x1xself.hidden_size大小的张量
        return torch.zeros(size=(1, 1, self.hidden_size), device=device)


# todo: 5-构建解码器类
class AttnDecoderRNN(nn.Module):
    # todo: 5-1 构造方法, 搭建神经网络模型结构
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # max_length    最大长度10
        super(AttnDecoderRNN, self).__init__()
        # 解码器 词嵌入层单词数 eg：4345
        # 也是最终输出层的类别数, 输出维度 -> 生成式模型, 预测下一个词, 词表大小=概率矩阵
        self.output_size = output_size
        # 词嵌入层每个单词的特征数/初始法文词向量 eg:256
        # 偷懒: 也是gru层的隐藏特征数
        self.hidden_size = hidden_size
        # 失活概率
        self.dropout_p = dropout_p
        # 所有样本中句子最大长度为10, 进行统一
        self.max_length = max_length

        # 定义nn.Embedding层 nn.Embedding(4345,256)
        # num_embeddings: 法文词表大小
        self.embedding = nn.Embedding(num_embeddings=self.output_size,
                                      embedding_dim=self.hidden_size)

        # 定义线性层1：求q的注意力权重分布
        # 查询张量Q: 解码器每个时间步的隐藏层输出或者是当前输入的x
        # 键张量K: 解码器上一个时间步的隐藏层输出
        # self.hidden_size * 2 = q + k
        # self.max_length: 长度不为最大长度10, 后续用0进行填充, 对应位置的权重值为0
        # 输出形状 (1, 1, 10) 和 v形状(1, 10, 256) 进行三维矩阵乘法 ->c的形状(1, 1, 256)
        self.attn = nn.Linear(in_features=self.hidden_size * 2, out_features=self.max_length)

        # 定义线性层2：q+注意力结果表示融合后，在按照指定维度输出
        # 值张量V:编码部分每个时间步输出结果组合而成
        # self.hidden_size * 2 = q + v
        self.attn_combine = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size)

        # 定义dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)

        # 定义gru层
        self.gru = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          batch_first=True)

        # 全连接层
        # 定义out层 解码器按照类别进行输出(256,4345)
        # in_features: 上一层gru层的输出
        # out_features: 预测的类别数=法文词表大小
        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=self.output_size)

        # 实例化softomax层 数值归一化 以便分类
        # 方式1: LogSoftmax+NLLLoss组合计算损失
        # 方式2: 直接使用CrossEntropyLoss损失函数, 内部包含log+softmax计算
        self.softmax = nn.LogSoftmax(dim=-1)

    # todo: 5-1 前向传播方法, 预测下一个词, 解码器一定是一个词一个词预测
    def forward(self, input, hidden, encoder_outputs):
        """
        预测下一个词
        :param input: 一个词表示 (句子数, 1)  q
        :param hidden: k  第1个时间步的k就是编码器hn
        :param encoder_outputs:  v 编码器output 原c
        :return:
        """
        print("input--->", input.shape, input)
        print("hidden--->", hidden.shape, hidden)
        print("encoder_outputs--->", encoder_outputs.shape, encoder_outputs)
        # input代表q [1,1] 二维数据 hidden代表k [1,1,256] encoder_outputs代表v [1,10,256]

        # 数据经过词嵌入层
        # 数据形状 [1,1] --> [1,1,256]
        embedded = self.embedding(input)
        print("embedded--->", embedded.shape, embedded)

        # 使用dropout进行随机丢弃，防止过拟合
        embedded = self.dropout(embedded)
        print("embedded--->", embedded.shape, embedded)

        # 1 求查询张量q的注意力权重分布, attn_weights[1,1,10]
        # q和k在特征维度拼接 + 线性层计算 -> 权重分数
        # self.attn(torch.cat(tensors=(embedded, hidden), dim=-1)) -> 权重分数
        # softmax->权重概率
        attn_weights = torch.softmax(
            self.attn(torch.cat(tensors=(embedded, hidden), dim=-1)), dim=-1)
        print("attn_weights--->", attn_weights.shape, attn_weights)

        # 2 求查询张量q的注意力结果表示 bmm运算, attn_applied[1,1,256]
        # 计算动态c, 带权重的c
        # attn_weights: 权重概率
        # encoder_outputs: 原c
        # [1,1,10], [1,10,256] ---> [1,1,256]
        attn_applied = torch.bmm(input=attn_weights, mat2=encoder_outputs)
        print("attn_applied--->", attn_applied.shape, attn_applied)

        # 3 q 与 attn_applied 融合，[1,1,512]
        # q和动态c融合, 得到下一层gru的输入表示
        # q和动态c特征维度拼接 + 线性层计算
        # [1, 1, 256] + [1, 1, 256] = [1, 1, 512]
        q_c_cat = torch.cat(tensors=(embedded, attn_applied), dim=-1)
        # 再按照指定维度输出 output[1,1,256], gru层输入形状要求
        gru_input = self.attn_combine(q_c_cat)
        print("gru_input--->", gru_input.shape, gru_input)

        # 查询张量q的注意力结果表示 使用relu激活
        gru_input = torch.relu(gru_input)

        # 查询张量经过gru、softmax进行分类结果输出
        # 数据形状[1,1,256],[1,1,256] --> [1,1,256], [1,1,256]
        output, hidden = self.gru(gru_input, hidden)
        print("output--->", output.shape, output)

        # output经过全连接层 out+softmax层, 全连接层要求输入数据为二维数据
        # 数据形状[1,1,256]->[1,256]->[1,4345]
        # print("output[:, 0, :]--->", output[:, 0, :].shape, output[:, 0, :])  # (句子数, 隐藏维度数), 以二维数据传入到线性层
        output = self.softmax(self.out(output[:, 0, :]))

        # print("output2--->", output.shape, output
        # )
        # 返回解码器分类output[1,4345]，最后隐层张量hidden[1,1,256] 注意力权重张量attn_weights[1,1,10]
        return output, hidden, attn_weights


if __name__ == '__main__':
    # normalizeString("I am a boy@. I am a man.\n")
    # 加载数据集到内存中, 并构建词表

    (english_word2index, english_index2word, english_word_n,
     french_word2index, french_index2word, french_word_n,
     my_pairs,) = my_getdata(data_path='data/eng-fra-v2.txt')

    # 构建张量数据集对象
    my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
    print("my_dataset--->", my_dataset)
    print("my_dataset的样本条目数--->", len(my_dataset))
    # 对象名[下标] 或者 传入到数据加载器中遍历时 都会调用 __getitem__魔法方法
    # print(my_dataset[0])
    # 构建数据加载器对象
    # dataset: 张量数据集对象 MyPairsDataset类的对象
    # shuffle: 是否打乱数据
    # batch_size: 批次大小, 目前值只能为1, 因为每个句子的长度不一致, 后续我们会实现句子长度规范
    # drop_last: 是否丢弃最后一个批次数据, 如果不能整除一般就丢弃

    my_dataloader = DataLoader(dataset=my_dataset,
                               shuffle=True,
                               batch_size=1,
                               drop_last=True)
    # print('my_dataloader--->', my_dataloader)

    # 实例化编码器模型对象  对象名=类名(参数1, 参数2, ...)
    # english_word_n: 英文词表大小
    # word_dim: 词嵌入维度 512
    # hidden_size: gru层的特征维度 256
    # to(device): 将模型迁移到对应的设备上 GPU/CPU 默认CPU
    encoder = EncoderRNN(input_size=english_word_n, word_dim=512, hidden_size=256).to(device)
    print('encoder--->', encoder)
    # # 获取encoder模型中embedding层  ->返回的词嵌入层对象
    # print('encoder.embedding--->', encoder.embedding)
    # # 获取encoder模型中embedding层的weights和bias
    # print('encoder.embedding.weight--->', encoder.embedding.weight)
    # print('encoder.gru.bias--->', encoder.gru.bias)
    # print('================================')
    # print('encoder.gru.all_weights--->', encoder.gru.all_weights)

    # 实例化解码器模型对象
    # output_size: 法文词表大小
    # hidden_size: embedding层&gru层的特征维度  当前示例相等
    # max_length: 句子最大长度, 文本长度规范
    decoder = AttnDecoderRNN(output_size=french_word_n,
                             hidden_size=256,
                             dropout_p=0.1,
                             max_length=MAX_LENGTH).to(device)
    print('decoder--->', decoder)

    # 训练遍历数据加载, 会自动调用my_dataset中的__getitem__魔法方法
    for x, y in my_dataloader:
        print('x.shape--->', x.shape, x)
        print('y.shape--->', y.shape, y)
        # 调用编码器模型进行前向传播  一批样本一批样本进行encoder处理
        # encoder.inithidden(): 对象名.方法名() 初始化h0
        output, hn = encoder(input=x, h0=encoder.inithidden())

        # 准备decoder模型需要的参数 q k v
        # q: 第一个词一定是起始符号, 后续在模型训练时实现, 当前就在y中第一个词表示
        # k: 编码器的hn

        # v: 编码器的output, 需要对 v 进行长度规范, 每个句子的长度不一致, 统一为max_length, 不足的填充0
        # 全0的表示(1,10,256), 后续将真实的encoder的output添加进来
        encoder_output_c = torch.zeros(
            1, MAX_LENGTH, encoder.hidden_size, device=device)
        # print('encoder_output_c--->', encoder_output_c.shape, encoder_output_c)
        # print('output.shape[1]--->', output.shape[1])  # 编码器output的句子长度
        for idx in range(output.shape[1]):
            # 循环中将每个时间步的输出值赋值给中间语义张量C
            # encode_output_c->(1, 10, 256)
            # output->(1, 句子长度, 256) output[:, idx, :]->所有句子第idx词语的语义向量
            # 长度不为10的句子, 其他位置就是全0
            encoder_output_c[:, idx, :] = output[:, idx, :]
        # print('encoder_output_c--->', encoder_output_c.shape, encoder_output_c)

        # 解码器进行解码处理, 一定是一个词一个词进行解码
        print('y.shape[1]--->', y.shape[1])  # 法文句子的长度, 最多预测法文句子长度的次数
        for i in range(y.shape[1]):
            # 获取第i个词
            # y[:, i] -> 一维向量, 但是embedding层接收二维向量表示
            tmp_x = y[:, i].reshape(-1, 1)  # (句子数, 1)
            print('tmp_x--->', tmp_x)
            # 调用解码器模型进行前向传播
            # tmp_x: q, 当前模拟, 第一个词没有传入其实标识符, 后续模型训练时第1个词再传入起始标识符
            # hn: k
            # encoder_output_c: v
            output, hidden, attn_weights = decoder(tmp_x, hn, encoder_output_c)
        break
