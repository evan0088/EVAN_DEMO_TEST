"""
需求：基于train.txt中的数据 ①构建(x,y)样本对 ②构建word2id(tokenizer)
思路步骤：
1. 初始化数据集
    1.1 加载配置文件
    1.2.基于深度学习实现NER 初始化词表
    1.3 加载train.txt
2.基于深度学习实现NER. 基于标点符号分割句子，分割(x,y)样本
3. 构建词表数据，并转换成id
    3.1 构建词表，得到index2word
    3.2.基于深度学习实现NER 构建word2index (word2id)
    3.3 保存词表
"""

from config import *
# 1. 初始化数据集
conf = Config()

'''构造数据集'''

def build_data():
    datas = []
    sample_x = []
    sample_y = []
    vocab_list = ["PAD", 'UNK']
    for line in open(conf.train_path, 'r', encoding='utf-8'):
        # TODO 别忘了做strip
        line = line.rstrip().split('\t')
        if not line:
            continue
        char = line[0]
        if not char:
            continue
        cate = line[-1]
        sample_x.append(char)
        sample_y.append(cate)
        #  3.1 构建词表，得到index2word
        if char not in vocab_list:
            vocab_list.append(char)
        # 2.基于深度学习实现NER. 基于标点符号分割句子，分割(x,y)样本
        if char in ['。', '?', '!', '！', '？']:
            datas.append([sample_x, sample_y])
            sample_x = []
            sample_y = []
    #  3.2.基于深度学习实现NER 构建word2index (word2id)
    word2id = {wd: index for index, wd in enumerate(vocab_list)}
    #  3.3 保存词表
    write_file(vocab_list, conf.vocab_path)
    return datas, word2id


'''保存字典文件'''


def write_file(wordlist, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(wordlist))


if __name__ == '__main__':
    datas, word2id = build_data()
    # 每个句子的字符数：260000/7836=33.1801939765
    print(len(datas))
    print(datas[:4])
    print(word2id)
    # 1759
    # 260000/1759=147.811256396
    print(len(word2id))
