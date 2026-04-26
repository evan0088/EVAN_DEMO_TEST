import torch
from tensorflow.keras.preprocessing import sequence
from torch.nn.utils.rnn import pad_sequence

# cutlen根据数据分析中句子长度分布，覆盖90%左右语料的最短长度.
# 这里假定cutlen为8
cutlen = 8


def padding_truncating(x_train):
    # 使用sequence.pad_sequences即可完成
    # truncating:pre->表示从前面截断 默认 post->表示从后面截断
    # padding:pre->表示从前面补齐 默认 post->表示从后面补齐
    return sequence.pad_sequences(sequences=x_train, maxlen=cutlen, truncating='pre', padding='post')


def demo02():
    # 形状(句子长度, 词维度)
    a = torch.ones(5, 10)
    b = torch.ones(7, 10)
    c = torch.ones(8, 10)
    # 根据批次中的最大长度进行填充
    # batch_first: 默认为False, 返回形状为(句子长度, 句子数, 词维度); True返回形状为(句子数, 句子长度, 词维度)
    ret = pad_sequence(sequences=[a, b, c], batch_first=True)
    print('ret--->', ret.shape, ret)


if __name__ == '__main__':
    x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
               [2, 32, 1, 23, 1]]

    res = padding_truncating(x_train)
    print(res)

    demo02()
