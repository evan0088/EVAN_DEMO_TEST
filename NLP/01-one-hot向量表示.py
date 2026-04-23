# 导入keras中的词汇映射器Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
# 导入用于对象保存与加载的joblib
import joblib

def one_hot_vector():
    # todo:1- 初始化词表
    vocabs = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]

    # todo:2- 实例化tokenizer对象, 分词器对象
    tokenizer = Tokenizer()
    print('tokenizer--->', tokenizer)
    # 构建词表, 词和词下标索引之间的映射关系
    tokenizer.fit_on_texts(texts=vocabs)
    print('tokenizer.word2index--->', tokenizer.word_index)
    print('tokenizer.index2word--->', tokenizer.index_word)

    # todo:3-构建one-hot词向量表示
    # one_hot_vector = tokenizer.texts_to_matrix(texts=vocabs, mode='binary')
    one_hot_vector = tokenizer.texts_to_matrix(texts=vocabs, mode='binary')[:,  1:]
    print('one_hot_vector--->', one_hot_vector)

    # todo:4-将词和one-hot词向量进行对应
    for word, vector in zip(vocabs, one_hot_vector):
        result = vector.astype(int).tolist()
        print(word, result)

    # todo:4-存储分词器对象的结果
    joblib.dump(tokenizer, './onehot_tokenizer.joblib')

    # todo:5-加载分词器对象, 进行one-hot转换
    tokenizer = joblib.load('./onehot_tokenizer.joblib')
    result = tokenizer.texts_to_matrix(texts=['王力宏'], mode='binary')[0,1:]
    print('result--->', result)


if __name__ == '__main__':
    one_hot_vector()