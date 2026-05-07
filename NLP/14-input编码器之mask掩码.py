import torch
import matplotlib.pyplot as plt


# 下三角：全1三角在下, 0代表掩码
# 下三角矩阵作用: 生成字符时,希望模型不要使用当前字符后面的字符。
# 使用遮掩mask，防止未来的信息可能被提前利用
# 实现方法：下三角矩阵
# 函数 subsequent_mask 实现分析
# 产生上三角矩阵 torch.triu(torch.ones((size, size))).type(torch.uint8)
# 产生下三角矩阵 torch.tril(torch.ones((size, size))).type(torch.uint8)
# 自回归掩码
def subsequent_mask(size):
    # 产生下三角矩阵 产生一个方阵
    # 词数的方阵->对attn_weights进行掩码 维度就是词数*词数
    subsequent_mask = torch.tril(torch.ones((size, size))).type(torch.uint8)
    return subsequent_mask


# 填充掩码
def padding_mask():
    # 文本词下标表示
    # 句子2, 第3和4个词填充0  0->词表中第1个词的下标
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 0, 0]])
    print(x!=0)
    print((x != 0).type(torch.uint8))
    padding_mask = (x != 0).type(torch.uint8)

    scores = torch.randn(2, 4)
    scores = scores.masked_fill(padding_mask==0, float('-inf'))
    print('scores--->', scores)


if __name__ == '__main__':
    # 产生5*5的下三角矩阵
    size = 5
    mask = subsequent_mask(size)
    print('下三角矩阵--->\n', mask)

    # 掩码张量可视化
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20))
    plt.show()

    # 因果掩码操作
    # 模拟注意力分数矩阵, 词和词之间运算的方阵结果
    scores = torch.randn(5, 5)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    print('scores--->', scores)

    # 填充掩码
    padding_mask()