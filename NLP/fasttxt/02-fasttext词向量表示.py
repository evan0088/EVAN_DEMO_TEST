# 导入fasttext
import fasttext


# 使用fasttext的train_unsupervised(无监督训练方法)进行词向量的训练
# 它的参数是数据集的持久化文件路径'data/fil9'
# 注意，该行代码执行耗时很长
# model = fasttext.train_unsupervised('data/fil9')
# 设置超参数
# model = fasttext.train_unsupervised('data/fil9', "cbow", dim=300, epoch=1, lr=0.1, thread=8)
# 保存向量模型
# model.save_model("data/fil9.bin")

# 可以使用以下代码加载文本预处理章节已经训练好的模型
model = fasttext.load_model("data/fil9.bin")
print('model--->', model)

# 通过get_word_vector方法来获得指定词汇的词向量
print(model.get_word_vector("the"))
print(len(model.get_word_vector("the")))
# 获取句子向量表示
print(model.get_sentence_vector("the quick brown fox jumps over the lazy dog"))

# 获取相关的词的向量表示, 使用特定数据集训练的模型
# k: 返回最相关的前k个词
print(model.get_nearest_neighbors("周杰伦", k=5))


# 词向量迁移, 使用已经训练好的向量模型直接对文本进行向量转换
# 加载预训练模型
model = fasttext.load_model("data/cc.zh.300.bin")
print(model.get_sentence_vector("机器学习"))
print(model.get_nearest_neighbors('七夕', k=5))