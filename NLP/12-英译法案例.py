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


# todo: 文本清洗工具函数
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


# todo: 加载数据集到内存中, 以及构建英文词表和法文词表
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


# todo: 构建张量数据集对象
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



if __name__ == '__main__':
    # normalizeString("I am a boy@. I am a man.\n")
    # 加载数据集到内存中, 并构建词表
    (english_word2index, english_index2word,english_word_n,
     french_word2index,french_index2word,french_word_n,my_pairs,)= my_getdata(data_path='data/eng-fra-v2.txt')
    # 构建张量数据集对象
    my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
    print("my_dataset--->", my_dataset)
    print("my_dataset的样本条目数--->", len(my_dataset))
    # 对象名[下标] 或者 传入到数据加载器中遍历时 都会调用 __getitem__魔法方法
    # print(my_dataset[0])
    # exit()
    # 构建数据加载器对象
    # dataset: 张量数据集对象 MyPairsDataset类的对象
    # shuffle: 是否打乱数据
    # batch_size: 批次大小, 目前值只能为1, 因为每个句子的长度不一致, 后续我们会实现句子长度规范
    # drop_last: 是否丢弃最后一个批次数据, 如果不能整除一般就丢弃
    my_dataloader = DataLoader(dataset=my_dataset,
                               shuffle=True,
                               batch_size=1,
                               drop_last=True)
    print('my_dataloader--->', my_dataloader)

    # 训练遍历数据加载, 会自动调用my_dataset中的__getitem__魔法方法
    for x, y in my_dataloader:
        print('x.shape--->', x.shape, x)
        print('y.shape--->', y.shape, y)
        # print([english_index2word[int(x)] for x in x[0]])
        break