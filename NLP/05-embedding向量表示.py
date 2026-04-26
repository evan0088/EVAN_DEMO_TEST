import torch
import torch.nn as nn
import jieba


# todo:1-初始化文本
text = "我爱自然语言处理, 我爱看足球"

# todo:2-实例化embedding层对象
# num_embeddings: 词表大小, 词表中有多少个词, 当前就设置多少
# embedding_dim: 词向量的维度数, 自定义, 10, 20, 100
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=5)
print('embedding--->', embedding)

print('初始权重:', embedding.weight.data)
print('是否需要梯度:', embedding.weight.requires_grad)

# todo:3-使用embedding层进行文本向量化
word_list = jieba.lcut(text)
print('word_list--->', word_list)
# 获取词对应的词下标
word_index = [word_list.index(word) for word in word_list]
print('word_index--->', word_index)
# 将词下标列表转换成张量对象
# embedding 只能接受张量对象作为输入
word_index_tensor = torch.tensor(word_index)
print('word_index_tensor--->', word_index_tensor)
# 调用embedding层对象进行向量化
result = embedding(word_index_tensor)
print('result--->', result.shape, result)
