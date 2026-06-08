"""
1. 加载数据
2. label
3. 处理input_ids
4. 处理位置编码
5. 构建Dataset
6. dataloader
"""

"""
需求：处理关系数据及相关文本数据，进行编码、格式转换、数据填充等操作，为模型训练做准备
思路步骤：
1. 准备工作:
    1.1 从配置文件导入配置信息，初始化关系类型字典relation2id
    1.2 读取配置文件中的关系数据路径，逐行读取关系数据文件，将关系类型及其对应的id存入relation2id字典
2. 编码训练、测试数据集格式：
    2.1 初始化数据列表、标签列表、实体位置列表、实体列表和关系计数字典
    2.2 读取数据文件，处理每行数据：
        2.2.1 过滤不在计数字典中的关系及计数超过2000的关系
        2.2.2 记录实体信息，获取实体在句子中的位置
        2.2.3 记录句子、标签、实体位置等信息到相应列表
    2.3 返回处理后的数据
3. 获取单词与id的映射：
    3.1 调用get_txt_data获取训练数据
    3.2 构建包含去重单词的词汇表
    3.3 生成单词到id和id到单词的映射字典并返回
4. 将句子转为id形式并补全长度：
    4.1 遍历句子中的单词，将其转为id，不存在则用'UNKNOW'的id
    4.2 若id列表长度达到最大长度则截取，否则用'BLANK'的id补齐
5. 转换实体位置信息：将实体位置信息进行转换，确保不出现负数
6. 将位置信息转为id形式并补全长度：
    6.1 对位置信息进行转换
    6.2 若转换后的位置id列表长度达到最大长度则截取，否则用142补齐
"""

# coding:utf-8
from config import *
from itertools import chain
from collections import Counter

conf = Config()
# 获取关系类型字典
relation2id = {}
with open(conf.rel_data_path, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        word, id = line.rstrip().split(' ')
        if word not in relation2id:
            relation2id[word] = id


def sent_padding(words, word2id):
    """把句子 words 转为 id 形式，并自动补全为 max_len 长度。"""
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id['UNKNOW'])
    if len(ids) >= conf.max_len:
        return ids[:conf.max_len]
    ids.extend([word2id['BLANK']] * (conf.max_len - len(ids)))
    return ids


def pos(num):
    '''
    将实体位置信息进行转换，因为pos_embedding不能出现负数
    '''
    if num < -70:
        return 0
    if num >= -70 and num <= 70:
        return num + 70
    if num > 70:
        return 142


def position_padding(pos_ids):
    '''
    """把 pos位置信息 转为 id 形式，并自动补全为 max_len 长度。"""
    '''
    pos_ids = [pos(id) for id in pos_ids]
    if len(pos_ids) >= conf.max_len:
        return pos_ids[:conf.max_len]
    pos_ids.extend([142] * (conf.max_len - len(pos_ids)))
    return pos_ids


def get_txt_data(data_path):
    '''
    编码训练、测试数据集格式
    '''
    datas = []
    labels = []
    positionE1 = []
    positionE2 = []
    entities = []
    count_dict = {key: 0 for key, value in relation2id.items()}
    with open(data_path, 'r', encoding='utf-8') as tfr:
        for line in tfr.readlines():
            line = line.rstrip().split(' ', maxsplit=3)
            if line[2] not in count_dict:
                continue
            if count_dict[line[2]] > 2000:
                continue
            else:
                entities.append([line[0], line[1]])
                sentence = []
                index1 = line[3].index(line[0])
                position1 = []
                index2 = line[3].index(line[1])
                position2 = []
                assert len(line) == 4
                for i, word in enumerate(line[3]):
                    sentence.append(word)
                    position1.append(i - index1)
                    position2.append(i - index2)

                datas.append(sentence)
                labels.append(relation2id[line[2]])
                positionE1.append(position1)
                positionE2.append(position2)
                count_dict[line[2]] += 1

    return datas, labels, positionE1, positionE2, entities


def get_word_id(data_path):
    '''
    文本数字化表示处理，得到word2id, id2word
    '''
    datas, labels, positionE1, positionE2, entities = get_txt_data(data_path)
    data_list = list(set(chain(*datas)))
    word2id = {word: id for id, word in enumerate(data_list)}
    id2word = {id: word for id, word in enumerate(data_list)}
    # len = 3468  0~3467
    word2id["BLANK"] = len(word2id)
    # len = 3469 0~3467
    word2id["UNKNOW"] = len(word2id)
    # len = 3470
    id2word[len(id2word)] = "BLANK"
    id2word[len(id2word)] = "UNKNOW"
    return word2id, id2word
