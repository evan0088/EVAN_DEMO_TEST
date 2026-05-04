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
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # input_size 编码器 词嵌入层单词数 eg：2803
        self.input_size = input_size
        # hidden_size 编码器 词嵌入层每个单词的特征数 eg: 256
        # self.word_dim = word_dim
        # 偷懒: 也是gru层的隐藏层特征数
        self.hidden_size = hidden_size  # 合二为一, 既是初始词向量维度又是gru层的特征维度

        # 实例化nn.Embedding层
        # num_embeddings: 词表大小
        # embedding_dim: 词嵌入维度/初始词向量维度
        self.embedding = nn.Embedding(
            num_embeddings=self.input_size, embedding_dim=self.hidden_size
        )

        # 实例化nn.GRU层 注意参数batch_first=True->(batch_size, seq_len, hidden_size)
        # input_size: 上一层的输出维度/embedding层的词向量维度
        # hidden_size: 隐藏层维度/gru层的特征维度
        self.gru = nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True
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
        # print("input--->", input.shape, input)
        # print("hidden--->", hidden.shape, hidden)
        # print("encoder_outputs--->", encoder_outputs.shape, encoder_outputs)
        # input代表q [1,1] 二维数据 hidden代表k [1,1,256] encoder_outputs代表v [1,10,256]

        # 数据经过词嵌入层
        # 数据形状 [1,1] --> [1,1,256]
        embedded = self.embedding(input)
        # print("embedded--->", embedded.shape, embedded)

        # 使用dropout进行随机丢弃，防止过拟合
        embedded = self.dropout(embedded)
        # print("embedded--->", embedded.shape, embedded)

        # 1 求查询张量q的注意力权重分布, attn_weights[1,1,10]
        # q和k在特征维度拼接 + 线性层计算 -> 权重分数
        # self.attn(torch.cat(tensors=(embedded, hidden), dim=-1)) -> 权重分数
        # softmax->权重概率
        attn_weights = torch.softmax(
            self.attn(torch.cat(tensors=(embedded, hidden), dim=-1)), dim=-1)
        # print("attn_weights--->", attn_weights.shape, attn_weights)

        # 2 求查询张量q的注意力结果表示 bmm运算, attn_applied[1,1,256]
        # 计算动态c, 带权重的c
        # attn_weights: 权重概率
        # encoder_outputs: 原c
        # [1,1,10], [1,10,256] ---> [1,1,256]
        attn_applied = torch.bmm(input=attn_weights, mat2=encoder_outputs)
        # print("attn_applied--->", attn_applied.shape, attn_applied)

        # 3 q 与 attn_applied 融合，[1,1,512]
        # q和动态c融合, 得到下一层gru的输入表示
        # q和动态c特征维度拼接 + 线性层计算
        q_c_cat = torch.cat(tensors=(embedded, attn_applied), dim=-1)
        # 再按照指定维度输出 output[1,1,256], gru层输入形状要求
        gru_input = self.attn_combine(q_c_cat)
        # print("gru_input--->", gru_input.shape, gru_input)

        # 查询张量q的注意力结果表示 使用relu激活
        gru_input = torch.relu(gru_input)

        # 查询张量经过gru、softmax进行分类结果输出
        # 数据形状[1,1,256],[1,1,256] --> [1,1,256], [1,1,256]
        output, hidden = self.gru(gru_input, hidden)
        # print("output--->", output.shape, output)

        # output经过全连接层 out+softmax层, 全连接层要求输入数据为二维数据
        # 数据形状[1,1,256]->[1,256]->[1,4345]
        # print("output[:, 0, :]--->", output[:, 0, :].shape, output[:, 0, :])  # (句子数, 隐藏维度数), 以二维数据传入到线性层
        output = self.softmax(self.out(output[:, 0, :]))

        # 返回解码器分类output[1,4345]，最后隐层张量hidden[1,1,256] 注意力权重张量attn_weights[1,1,10]
        return output, hidden, attn_weights


# todo: 测试函数
def test_decoder_attn():
    # normalizeString("I am a boy@. I am a man.\n")
    # 加载数据集到内存中, 并构建词表
    (english_word2index, english_index2word, english_word_n,
     french_word2index, french_index2word, french_word_n, my_pairs,) = my_getdata(data_path='data/eng-fra-v2.txt')

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
    # hidden_size: embedding层&gru层的特征维度  当前示例相等
    # to(device): 将模型迁移到对应的设备上 GPU/CPU 默认CPU
    encoder = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device)
    # print('encoder--->', encoder)
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
    # print('decoder--->', decoder)

    # 训练遍历数据加载, 会自动调用my_dataset中的__getitem__魔法方法
    for x, y in my_dataloader:
        # print('x.shape--->', x.shape, x)
        # print('y.shape--->', y.shape, y)
        # 调用编码器模型进行前向传播  一批样本一批样本进行encoder处理
        # encoder.inithidden(): 对象名.方法名() 初始化h0
        output, hn = encoder(input=x, h0=encoder.inithidden())
        # print('hn--->', hn.shape, hn)

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
        # print('y.shape[1]--->', y.shape[1])  # 法文句子的长度, 最多预测法文句子长度的次数
        # print('y--->', y)
        for i in range(y.shape[1]):
            # 获取第i个词
            # y[:, i] -> 一维向量, 但是embedding层接收二维向量表示
            tmp_x = y[:, i].reshape(-1, 1)  # (句子数, 1)
            # print('tmp_x--->', tmp_x)
            # 调用解码器模型进行前向传播
            # tmp_x: q, 当前模拟, 第一个词没有传入其实标识符, 后续模型训练时第1个词再传入起始标识符
            # hn: k
            # encoder_output_c: v
            output, hn, attn_weights = decoder(tmp_x, hn, encoder_output_c)
        break


# todo: 6-模型训练
# 准备训练参数
# 模型训练参数
mylr = 1e-4
epochs = 2
print_interval_num = 1000
plot_interval_num = 100


# todo:6-1 主函数训练
# 1.加载数据集
# 2.实例化数据加载器对象
# 3.实例化模型对象(编码器和解码器)
# 4.实例化优化器对象(编码器和解码器)
# 5.实例化损失函数对象
# 6.双层循环遍历进行模型训练, 外层循环遍历epoch轮次, 内层训练遍历数据加载器对象/batch/批次
# 7.保存模型
def train_seq2seq():
    # 获取数据
    (english_word2index, english_index2word, english_word_n,
     french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata(data_path='data/eng-fra-v2.txt')

    # 实例化 mypairsdataset对象
    mypairsdataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)

    # 实例化mydataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化编码器模型 my_encoderrnn
    my_encoderrnn = EncoderRNN(english_word_n, 256).to(device)
    # 实例化解码器模型 my_attndecoderrnn
    my_attndecoderrnn = AttnDecoderRNN(output_size=french_word_n, hidden_size=256, dropout_p=0.1, max_length=10).to(
        device)

    # 因为使用的encoder-decoder架构, 有两个模型, 都需要进行参数更新
    # 实例化编码器优化器 myadam_encode
    myadam_encode = optim.Adam(my_encoderrnn.parameters(), lr=mylr)
    # 实例化解码器优化器 myadam_decode
    myadam_decode = optim.Adam(my_attndecoderrnn.parameters(), lr=mylr)

    # 1个, 只需要对解码器的预测结果进行损失计算
    # 实例化损失函数 mycrossentropyloss = nn.NLLLoss()
    mynllloss = nn.NLLLoss()

    # 定义模型训练的变量
    plot_loss_list = []
    # 统计所有轮次的总批次数
    total_steps = epochs * len(mydataloader)
    # 当前累计批次数
    current_step = 0

    # 外层for循环 控制轮数 for epoch_idx in range(1, epochs + 1):
    for epoch_idx in range(1, epochs + 1):
        print_loss_total, plot_loss_total = 0.0, 0.0
        starttime = time.time()

        # 内层for循环 控制迭代次数 进行模型训练
        # start=1: 下标从1开始, 默认0, 数据开始从第1个开始取
        # item第1个值为1
        for item, (x, y) in enumerate(mydataloader, start=1):
            # 调用内部训练函数, 封装函数
            myloss = train_iters(x, y, my_encoderrnn, my_attndecoderrnn, myadam_encode, myadam_decode, mynllloss,
                                 total_steps, current_step)
            print_loss_total += myloss
            plot_loss_total += myloss
            # 累计训练批次数
            current_step += 1

            # 计算打印屏幕间隔损失-每隔1000次
            if item % print_interval_num == 0:
                print_loss_avg = print_loss_total / print_interval_num
                # 将总损失归0
                print_loss_total = 0
                # 打印日志，日志内容分别是：训练耗时，当前迭代步，当前进度百分比，当前平均损失
                print('轮次%d  损失%.6f 时间:%d' % (epoch_idx, print_loss_avg, time.time() - starttime))

            # 计算画图间隔损失-每隔100次
            if item % plot_interval_num == 0:
                # 通过总损失除以间隔得到平均损失
                plot_loss_avg = plot_loss_total / plot_interval_num
                # 将平均损失添加plot_loss_list列表中
                plot_loss_list.append(plot_loss_avg)
                # 总损失归0
                plot_loss_total = 0

        # 每个轮次保存模型
        torch.save(my_encoderrnn.state_dict(), 'model/my_encoderrnn_%d.pth' % epoch_idx)
        torch.save(my_attndecoderrnn.state_dict(), 'model/my_attndecoderrnn_%d.pth' % epoch_idx)

    # 所有轮次训练完毕 画损失图
    plt.figure()
    plt.plot(plot_loss_list.detach().numpy())
    plt.savefig('img/s2sq_loss.png')
    plt.show()


# todo:6-2 对内部训练代码逻辑封装成函数, 让代码更加清晰
def train_iters(
        x,
        y,
        my_encoderrnn: EncoderRNN,
        my_attndecoderrnn: AttnDecoderRNN,
        myadam_encode,
        myadam_decode,
        mynllloss,
        total_steps,
        current_step,
):
    """
    内部循环函数, 实现模型训练过程代码
    1. 编码器编码 -> 得到output和hn
    2. 解码器解码
    2.1 准备q k v, q->第1个值是SOS词下标 k->第1个值是编码器的hn v->编码器的output进行长度补齐的结果
    2.2 设置教师强制机制阈值  线性衰减策略  前期阈值大用更多真实y  后期阈值小用更多预测y
    2.3 一个时间步一个时间步解码
    2.3.1 调用解码器模型进行预测, 得到预测y
    2.3.2 预测y和真实y计算损失值
    2.3.3 根据教师强制机制选择下一个时间步使用预测y还是真实y
    2.3.4 梯度清零, 反向传播, 参数更新
    :param x: 英文句子
    :param y: 法文句子
    :param my_encoderrnn: 编码器模型
    :param my_attndecoderrnn: 解码器模型
    :param myadam_encode: 编码器优化器
    :param myadam_decode: 解码器优化器
    :param mynllloss: 损失函数对象
    :param total_steps: 总的批次数
    :param current_step: 当前的训练批次数
    :return: 平均损失
    """
    # 0 切换模型训练模式
    my_encoderrnn.train()
    my_attndecoderrnn.train()
    ########################################## 编码器模型进行编码操作 #########################################
    # 1 编码 encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)
    # 初始化h0 全0
    encode_hidden = my_encoderrnn.inithidden()
    encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)  # 一次性送数据
    # [1,6],[1,1,256] --> [1,6,256],[1,1,256]
    # print("encode_output--->", encode_output.shape, encode_output)
    # print("encode_hidden--->", encode_hidden.shape, encode_hidden)

    ########################################## 解码器模型进行解码操作 #########################################
    # 2 解码参数准备和解码
    # 解码参数1:v encode_output_c [1, 10,256]
    # 对编码器输出的encode_output进行文本长度规范, 统一长度为10

    # 创建一个全零矩阵 encode_output_c [1, 10, 256]
    encode_output_c = torch.zeros(
        1, MAX_LENGTH, my_encoderrnn.hidden_size, device=device
    )
    # 将编码器输出encode_ouput对应位置赋值给encode_output_c
    for idx in range(x.shape[1]):
        encode_output_c[:, idx, :] = encode_output[:, idx, :]
    # print("encode_output_c--->", encode_output_c.shape, encode_output_c)

    # 解码参数2: k 初始k就是编码器最后一个时间步的隐藏状态值
    decode_hidden = encode_hidden
    # print("decode_hidden--->", decode_hidden.shape, decode_hidden)

    # 解码参数3: q 使用的真实y或预测y下标表示, 第一个时间步的q是起始标识符词下标表示  也可以使用上一个时间步的隐藏状态值
    input_y = torch.tensor([[SOS_token]], device=device)
    # print("input_y--->", input_y.shape, input_y)

    # 初始化变量
    # 初始损失值
    myloss = 0.0
    # 初始词数
    iters_num = 0
    # 真实y的词数量/长度
    y_len = y.shape[1]

    # 教师强制机制, 阈值线性衰减
    teacher_forcing_ratio = max(0.1, 1 - (current_step / total_steps))
    # 阈值指数衰减
    # teacher_forcing_ratio = 0.9 ** current_step
    # True: 使用真实y
    # False: 使用预测y
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # print("use_teacher_forcing--->", use_teacher_forcing)

    # 循环遍历进行解码
    for idx in range(y_len):  # 停止条件1: 遍历次数达到真实y的长度, 模型处理的最大长度
        # 数据形状 [1,1],[1,1,256],[1,10,256] ---> [1,4345],[1,1,256],[1,1,10]
        output_y, decode_hidden, attn_weight = my_attndecoderrnn(
            input_y, decode_hidden, encode_output_c
        )
        # output_y: 预测y (样本数, 4345)
        # print("output_y--->", output_y.shape, output_y)
        # 获取真实y中当前idx时间步的真实词
        target_y = y[:, idx]
        # print("y--->", y.shape, y)
        # print("target_y--->", target_y.shape, target_y)
        myloss = myloss + mynllloss(output_y, target_y)
        # print("myloss--->", myloss)
        iters_num += 1
        # 使用teacher_forcing
        if use_teacher_forcing:
            # 获取真实样本作为下一个输入
            # reshape(shape=(-1, 1)): 模型要求的输入形状二维 (句子数, 1)  经过embedding变成三维
            input_y = y[:, idx].reshape(shape=(-1, 1))
            # print("input_y--->", input_y.shape, input_y)
        # 不使用teacher_forcing
        else:
            # 获取最大值的值和索引
            # topi: 最大概率的索引=词表的词下标
            topv, topi = output_y.topk(1)
            # 停止条件2:预测的y值是结束符号词下标表示
            if topi.item() == EOS_token:
                break
            # 获取预测y值作为下一个输入
            input_y = topi.detach()
            # print("input_y--->", input_y.shape, input_y)

    # 梯度清零
    myadam_encode.zero_grad()
    myadam_decode.zero_grad()

    # 反向传播, 当前批次的所有句子预测完后
    myloss.backward()

    # 梯度更新
    myadam_encode.step()
    myadam_decode.step()

    # 计算迭代次数的平均损失
    return myloss.item() / iters_num


# todo:7- 模型推理
# 主函数 推理封装函数
PATH1 = "model/my_encoderrnn_2.pth"
PATH2 = "model/my_attndecoderrnn_2.pth"


def dm_test_seq2seq_evaluate():
    # 加载数据集
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = my_getdata(data_path='data/eng-fra-v2.txt')

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_encoderrnn = EncoderRNN(input_size, hidden_size).to(device)

    """
    torch.load(map_location=)
    map_location: 指定如何重映射模型权重的存储设备（如 GPU → CPU 或 GPU → 其他 GPU）。
    # 加载到 CPU：map_location=torch.device('cpu') 或 map_location='cpu'。
    自动选择可用设备：map_location=torch.device('cuda')。
    自定义映射逻辑：通过函数定义设备映射规则。
    map_location=lambda storage, loc: storage -> 该lambda函数直接返回原始存储对象(storage)
    强制所有张量保留在保存时的设备上。当模型权重保存时的设备与当前环境一致时（例如均在CPU或同一GPU上），避免不必要的设备迁移。

    load_state_dict(strict=)
    strict:True（默认）:要求加载的权重键（keys）与当前模型的键完全匹配。如果存在不匹配（例如权重中缺少某些键，或模型有额外键），抛出RuntimeError。
    """
    my_encoderrnn.load_state_dict(
        torch.load(PATH1, map_location=lambda storage, loc: storage), strict=False
    )
    print("my_encoderrnn模型结构--->", my_encoderrnn)

    # 实例化模型
    input_size = french_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_attndecoderrnn = AttnDecoderRNN(input_size, hidden_size).to(device)
    # my_attndecoderrnn.load_state_dict(torch.load(PATH2))
    my_attndecoderrnn.load_state_dict(
        torch.load(PATH2, map_location=lambda storage, loc: storage), False
    )
    print("my_decoderrnn模型结构--->", my_attndecoderrnn)

    # 测试样本 [[英文, 法文], [英文, 法文], ...]
    my_samplepairs = [
        [
            "i m impressed with your french .",
            "je suis impressionne par votre francais .",
        ],
        ["i m more than a friend .", "je suis plus qu une amie ."],
        ["she is beautiful like her mother .", "elle est belle comme sa mere ."],
    ]
    print("my_samplepairs--->", len(my_samplepairs))

    for index, pair in enumerate(my_samplepairs):
        # 获取当前样本中的 英文句子和法文句子
        x = pair[0]
        y = pair[1]

        # 样本x 文本数值化
        tmpx = [english_word2index[word] for word in x.split(" ")]
        tmpx.append(EOS_token)
        # 转换成张量对象
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)
        # print("tensor_x--->", tensor_x)

        # 模型预测/推理
        # 对推理过程进行二次封装
        decoded_words, attentions = seq2seq_evaluate(
            tensor_x, my_encoderrnn, my_attndecoderrnn, french_index2word
        )
        # print("attentions--->", attentions)
        # [预测词1, 预测词2, ...]
        # print('decoded_words->', decoded_words)
        output_sentence = " ".join(decoded_words)

        print("\n")
        print(">", x)
        print("=", y)
        print("<", output_sentence)


# 模型评估代码与模型预测代码类似，需要注意使用with torch.no_grad()
# 模型预测时，第一个时间步使用SOS_token作为输入 后续时间步采用预测值作为输入，也就是自回归机制
def seq2seq_evaluate(
        x, my_encoderrnn: EncoderRNN, my_attndecoderrnn: AttnDecoderRNN, french_index2word
):
    # 上下文管理器, 不进行反向传播
    with torch.no_grad():
        # 模型评估模式
        my_encoderrnn.eval()
        my_attndecoderrnn.eval()

        ############################ 编码器编码 ############################
        # 1 编码：一次性的送数据
        encode_hidden = my_encoderrnn.inithidden()
        encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)

        ############################ 解码器解码 ############################
        # 2 解码参数准备
        # 解码参数1: v 固定长度中间语义张量c
        encoder_outputs_c = torch.zeros(
            1, MAX_LENGTH, my_encoderrnn.hidden_size, device=device
        )
        x_len = x.shape[1]
        # 补齐
        for idx in range(x_len):
            encoder_outputs_c[:, idx, :] = encode_output[:, idx, :]

        # 解码参数2: k 最后1个隐藏层的输出 作为 解码器的第1个时间步隐藏层输入
        decode_hidden = encode_hidden

        # 解码参数3: q 解码器第一个时间步起始符
        input_y = torch.tensor([[SOS_token]], device=device)

        # 3 自回归方式解码
        # 初始化预测的词汇列表
        decoded_words = []

        # 初始化attention张量
        decoder_attentions = torch.zeros(1, MAX_LENGTH, MAX_LENGTH)

        for idx in range(MAX_LENGTH):  # note:MAX_LENGTH=10
            # 解码器模型进行预测
            output_y, decode_hidden, attn_weights = my_attndecoderrnn(
                input_y, decode_hidden, encoder_outputs_c
            )
            # print("output_y--->", output_y.shape, output_y)
            # print("attn_weights--->", attn_weights.shape, attn_weights)
            # 预测值作为下一次时间步的输入值
            topv, topi = output_y.topk(1)
            decoder_attentions[:, idx, :] = attn_weights[:, 0, :]
            # print("decoder_attentions--->", decoder_attentions.shape, decoder_attentions)

            # 如果输出值是终止符，则循环停止
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(french_index2word[topi.item()])

            # 将本次预测的索引赋值给 input_y，进行下一个时间步预测
            # 推理时没有教师强制机制, 都是使用上一步预测的结果作为下一步的输入
            input_y = topi.detach()

    # 返回结果decoded_words，注意力张量权重分布表(把没有用到的部分切掉)
    # 句子长度最大是10, 长度不为10的句子的注意力张量其余位置为0, 去掉
    return decoded_words, decoder_attentions[:, :idx + 1, :]


# todo: 注意力权重矩阵绘图, 便于解释为什么当前位置预测出这个词
def dm_test_Attention():
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = my_getdata(data_path='data/eng-fra-v2.txt')

    # 实例化dataset对象
    mypairsdataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
    # 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_encoderrnn = EncoderRNN(input_size, hidden_size).to(device=device)
    # my_encoderrnn.load_state_dict(torch.load(PATH1))
    my_encoderrnn.load_state_dict(
        torch.load(PATH1, map_location=lambda storage, loc: storage), False
    )

    # 实例化模型
    input_size = french_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_attndecoderrnn = AttnDecoderRNN(input_size, hidden_size).to(device=device)
    # my_attndecoderrnn.load_state_dict(torch.load(PATH2))
    my_attndecoderrnn.load_state_dict(
        torch.load(PATH2, map_location=lambda storage, loc: storage), False
    )

    sentence = "we re both teachers ."
    # 样本x 文本数值化
    tmpx = [english_word2index[word] for word in sentence.split(" ")]
    tmpx.append(EOS_token)
    tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

    # 模型预测
    decoded_words, attentions = seq2seq_evaluate(
        tensor_x, my_encoderrnn, my_attndecoderrnn, french_index2word
    )
    print("decoded_words->", decoded_words)

    # print('\n')
    # print('英文', sentence)
    # print('法文', output_sentence)

    # 创建热图
    fig, ax = plt.subplots()
    # cmap:指定一个颜色映射，将数据值映射到颜色
    # viridis:从深紫色（低值）过渡到黄色（高值），具有良好的对比度和可读性
    cax = ax.matshow(attentions[0].cpu().detach().numpy(), cmap="viridis")
    # 添加颜色条
    fig.colorbar(cax)
    # 添加标签
    for (i, j), value in np.ndenumerate(attentions[0].cpu().detach().numpy()):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
    # 保存图像
    plt.savefig("img/s2s_attn.png")
    plt.show()

    print("attentions.numpy()--->\n", attentions.numpy())
    print("attentions.size--->", attentions.size())


if __name__ == '__main__':
    # test_decoder_attn()
    # train_seq2seq()
    # dm_test_seq2seq_evaluate()
    dm_test_Attention()
