# 定义LSTM的参数含义: (input_size/输入特征个数/词向量维度, hidden_size/隐层特征个数/隐层维度, num_layers/隐层层数)
# 定义输入张量x的参数含义: 默认->(sequence_length/句子长度, batch_size/句子数, input_size/词维度)
# 定义隐藏层初始张量h0和细胞初始状态张量c0的参数含义: (num_layers * num_directions/隐层层数, batch_size/句子数, hidden_size/隐层维度数)
# 隐层层数:如果单向就等于LSTM中的num_layers参数值; 如果双向就等于LSTM中的num_layers参数值*2
# 是否双向: bidirectional参数决定, 默认单向; True:双向
import torch.nn as nn
import torch


# TODO 单向LSTM
def dm01_lstm():
    # 创建LSTM层
    # 将rnn类换成lstm类
    lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1, batch_first=False)
    # 创建输入张量x 形状->(句子长度, 句子数, 词维度)
    input = torch.randn(size=(3, 3, 5))

    # c和h形状一致->(隐层数*方向, 句子数, 隐层维度)
    # 初始化隐藏状态
    h0 = torch.zeros(size=(1, 3, 6))
    # 初始化细胞状态
    c0 = torch.zeros(size=(1, 3, 6))

    # h和c是以数组形式传递和返回
    # hn输出两层隐藏状态, 最后1个隐藏状态值等于output输出值
    output, (hn, cn) = lstm(input, (h0, c0))
    print('output--->', output.shape, output)
    print('hn--->', hn.shape, hn)
    print('cn--->', cn.shape, cn)


# 双向LSTM
def dm02_lstm():
    # 创建LSTM层
    # 将rnn类换成lstm类
    lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1, batch_first=False, bidirectional=True)
    # 创建输入张量x 形状->(句子长度, 句子数, 词维度)
    input = torch.randn(size=(3, 3, 5))

    # c和h形状一致->(隐层数*方向, 句子数, 隐层维度)
    # 初始化隐藏状态
    h0 = torch.zeros(size=(1 * 2, 3, 6))
    # 初始化细胞状态
    c0 = torch.zeros(size=(1 * 2, 3, 6))

    # h和c是以数组形式传递和返回
    # hn输出两层隐藏状态, 最后1个隐藏状态值等于output输出值
    output, (hn, cn) = lstm(input, (h0, c0))
    print('output--->', output.shape, output)
    print('hn--->', hn.shape, hn)
    print('cn--->', cn.shape, cn)


if __name__ == '__main__':
    dm01_lstm()
    dm02_lstm()
