# 导入必备的工具包
import torch
# 导入Vocabulary，目的：用于构建, 存储和使用 `str` 到 `int` 的一一映射
# TODO fastNLP: 一款支持GPU的轻量级NLP库，支持ner和文本分类两种任务，也自带一些工具
from fastNLP import Vocabulary

# TODO  transformers的三层API： 1. pipeline：面向小白 2. autoModel: 稍微深入，但是不知道具体是什么模型类 3. SpecificModel: 特定模型，工作使用
from transformers import BertTokenizer

import json


# 构建配置文件Config类
class Config(object):
    def __init__(self):
        # 设置是否使用GPU来进行模型训练
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'mps'
        self.bert_path = r"D:\workspace\project\PYTHON\AI-Learning\知识图谱\code\4基于joint实现关系抽取\Casrel_RE\bert-base-chinese"
        self.num_rel = 18  # 关系的种类数
        self.batch_size = 8
        self.train_data_path = r"D:\workspace\project\PYTHON\AI-Learning\知识图谱\code\4基于joint实现关系抽取\Casrel_RE\data/train.json"
        self.dev_data_path = r"D:\workspace\project\PYTHON\AI-Learning\知识图谱\code\4基于joint实现关系抽取\Casrel_RE\data/dev.json"
        self.test_data_path = r"D:\workspace\project\PYTHON\AI-Learning\知识图谱\code\4基于joint实现关系抽取\Casrel_RE\data/test.json"
        self.rel_dict_path = r"D:\workspace\project\PYTHON\AI-Learning\知识图谱\code\4基于joint实现关系抽取\Casrel_RE\data/relation.json"
        id2rel = json.load(open(self.rel_dict_path, encoding='utf8'))
        self.rel_vocab = Vocabulary(padding=None, unknown=None)
        # vocab更新自己的字典，输入为list列表
        # TODO 这里存放就是 关系 -> ID, ID -> 关系的映射
        self.rel_vocab.add_word_lst(list(id2rel.values()))
        # TODO 加载bert的词表，得到 单词 -> ID, ID -> 单词的关系
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # TODO bert输出层微调适合这个参数
        self.learning_rate = 1e-5
        self.bert_dim = 768
        self.epochs = 10


if __name__ == '__main__':
    config = Config()
    print(config.rel_vocab)
