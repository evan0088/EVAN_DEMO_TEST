# x/output形状表示
# batch_first=Fase->形状默认(句子长度, 句子数, 词维度)
# batch_first=True->形状默认(句子数, 句子长度, 词维度)

# h->形状(隐藏层层数, 句子数, 隐层维度)

# 正常句子长度大于1, 隐藏层层数大于1, ouput!=hn
import torch
import torch.nn as nn


# todo:1-句子长度为1的输入
# 每个句子只有一个词组成  一个词代表一句话
def demo01():
    # 初始化x->(句子长度/词数, 句子数, 词维度)
    # 3个句子, 每个句子1个词, 词向量维度为5
    x = torch.randn(size=(1, 3, 5))

    # 实例化RNN层对象
    # input_size: 输入的词维度 或者 输入层的特征个数
    # hidden_size: rnn层的隐层维度 或者 隐藏层的特征个数
    # num_layers: 默认1层, rnn层数 或者 隐藏层层数
    # batch_first:
    # 默认False->输入和输出的形状为(句子长度, 句子数, 词维度)
    # True->输入和输出的形状为(句子数, 句子长度, 词维度)  后续transformer模型都是这种形状
    rnn = nn.GRU(input_size=5,
                 hidden_size=7,
                 num_layers=1,
                 batch_first=False)

    # 初始化h0(显性初始化), 如果不进行初始化, 默认初始化为0(隐性初始化)
    # 形状->(隐藏层层数, 句子数, 隐层维度)
    h0 = torch.zeros(size=(1, 3, 7))

    # 调用rnn层对象,生成output和hn
    # ouput, hn = rnn(x, h0)
    # 不设置h0, 隐性设置, 自动初始化为0
    ouput, hn = rnn(x)
    print('ouput--->', ouput.shape, ouput)
    print('hn--->', hn.shape, hn)
    # output=hn -> 因为句子中只有一个词, 并且只有一层隐藏层, 所以循环了一次


# todo:2-句子长度为3的输入
# 每个句子只有一个词组成  一个词代表一句话
def demo02():
    # 初始化x->(句子长度/词数, 句子数, 词维度)
    # 3个句子, 每个句子3个词, 词向量维度为5
    x = torch.randn(size=(3, 3, 5))

    # 实例化RNN层对象
    # input_size: 输入的词维度 或者 输入层的特征个数
    # hidden_size: rnn层的隐层维度 或者 隐藏层的特征个数
    # num_layers: 默认1层, rnn层数 或者 隐藏层层数
    # batch_first:
    # 默认False->输入和输出的形状为(句子长度, 句子数, 词维度)
    # True->输入和输出的形状为(句子数, 句子长度, 词维度)  后续transformer模型都是这种形状
    rnn = nn.GRU(input_size=5,
                 hidden_size=7,
                 num_layers=1,
                 batch_first=False)

    # 初始化h0(显性初始化), 如果不进行初始化, 默认初始化为0(隐性初始化)
    # 形状->(隐藏层层数, 句子数, 隐层维度)
    h0 = torch.zeros(size=(1, 3, 7))

    # 调用rnn层对象,生成output和hn
    # ouput, hn = rnn(x, h0)
    ouput, hn = rnn(x)
    print('ouput--->', ouput.shape, ouput)
    print('hn--->', hn.shape, hn)
    # output!=hn -> 因为句子中有3个词, 循环3次
    # output是h1 h2 h3 三个拼接结果concat, hn是最后一个时间步的隐藏状态值

    # -1: 0维度表示词数/句子长度, -1就是取每个句子的最后一个词
    last_h = ouput[-1, :, :]
    print('last_h--->', last_h.shape, last_h)


# 隐藏层层数大于1
def demo03():
    # 初始化x->(句子长度/词数, 句子数, 词维度)
    # 3个句子, 每个句子3个词, 词向量维度为5
    x = torch.randn(size=(3, 3, 5))

    # 实例化RNN层对象
    # input_size: 输入的词维度 或者 输入层的特征个数
    # hidden_size: rnn层的隐层维度 或者 隐藏层的特征个数
    # num_layers: 默认1层, rnn层数 或者 隐藏层层数
    # batch_first:
    # 默认False->输入和输出的形状为(句子长度, 句子数, 词维度)
    # True->输入和输出的形状为(句子数, 句子长度, 词维度)  后续transformer模型都是这种形状
    rnn = nn.GRU(input_size=5,
                 hidden_size=7,
                 num_layers=2,
                 batch_first=False)

    # 初始化h0(显性初始化), 如果不进行初始化, 默认初始化为0(隐性初始化)
    # 形状->(隐藏层层数, 句子数, 隐层维度)
    h0 = torch.zeros(size=(2, 3, 7))

    # 调用rnn层对象,生成output和hn
    ouput, hn = rnn(x, h0)
    # ouput, hn = rnn(x)
    print('ouput--->', ouput.shape, ouput)
    print('hn--->', hn.shape, hn)
    # ouput: 所有时间步最后一层隐层的状态值拼接结果
    # hn: 最后一个时间步所有层的隐藏状态值拼接结果


# 双向
def demo04():
    # 初始化x->(句子长度/词数, 句子数, 词维度)
    # 3个句子, 每个句子3个词, 词向量维度为5
    x = torch.randn(size=(3, 3, 5))

    # 实例化RNN层对象
    # input_size: 输入的词维度 或者 输入层的特征个数
    # hidden_size: rnn层的隐层维度 或者 隐藏层的特征个数
    # num_layers: 默认1层, rnn层数 或者 隐藏层层数
    # batch_first:
    # 默认False->输入和输出的形状为(句子长度, 句子数, 词维度)
    # True->输入和输出的形状为(句子数, 句子长度, 词维度)  后续transformer模型都是这种形状
    rnn = nn.GRU(input_size=5,
                 hidden_size=7,
                 num_layers=1,
                 batch_first=False,
                 bidirectional=True)

    # 初始化h0(显性初始化), 如果不进行初始化, 默认初始化为0(隐性初始化)
    # 形状->(隐藏层层数, 句子数, 隐层维度)
    h0 = torch.zeros(size=(1*2, 3, 7))

    # 调用rnn层对象,生成output和hn
    ouput, hn = rnn(x, h0)
    # ouput, hn = rnn(x)
    print('ouput--->', ouput.shape, ouput)
    print('hn--->', hn.shape, hn)
    # output!=hn -> 因为句子中有3个词, 循环3次
    # output是h1 h2 h3 三个拼接结果concat, hn是最后一个时间步的隐藏状态值

    # -1: 0维度表示词数/句子长度, -1就是取每个句子的最后一个词
    last_h = ouput[-1, :, :]
    print('last_h--->', last_h.shape, last_h)


if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
    demo04()