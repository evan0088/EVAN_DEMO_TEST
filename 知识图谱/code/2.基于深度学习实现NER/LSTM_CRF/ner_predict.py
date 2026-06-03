import torch.nn as nn
import torch.optim as optim
from model.ner_lstm import *
from model.ner_lstm_crf import *
from utils.data_loader import *
from tqdm import tqdm


"""
需求： 实现基于深度学习的实体抽取预测函数
思路步骤：
1. 加载模型和数据
    1.1 实例化模型，需要考虑具体用的是哪个模型
    1.2.基于深度学习实现NER 加载模型参数
    1.3 加载标签id换中文分类数据id2tag
2.基于深度学习实现NER. 处理预测数据
    2.基于深度学习实现NER.1 把输入的语料转成字符id列表
    2.基于深度学习实现NER.1 把字符id列表转成张量:x
    2.基于深度学习实现NER.2.基于深度学习实现NER 计算掩码张量:mask
3. 开启评估模式
    3.1 开启模型评估
    3.2.基于深度学习实现NER 设置不更新梯度
4. 前向传播
    4.1 把x, mask输入模型，得到预测值tag_id
        ① 对于BiLSTM, 得到多分类概率后转成tag_id
        ② 对于BiLSTM+CRF，直接得到tag_id
    4.2.基于深度学习实现NER 把tag_id转成分类名
5. 对前向传播数据做处理：实体抽取
    5.1 字符和label组成元组, 比如：('冠', 'B-DISEASE')
    5.2.基于深度学习实现NER 基于字符和label的BIO标签(B-, I- ,O)，拼接字符组成实体
        ① 如果是B-开头，则认为是一个实体的开始
        ② 如果是I-开头，则认为是一个实体的中间部分
        ③ 如果是O，则认为不是一个实体
    5.3 手动保存实体
    5.4 返回抽取出来的所有实体：以实体内容->实体类型格式

"""

# 实例化模型
models = {'BiLSTM': NERLSTM,
          'BiLSTM_CRF': NERLSTM_CRF}

model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)

if conf.model == 'BiLSTM':
    model.load_state_dict(torch.load('save_model/bilstm_best.pth'))
else:
    model.load_state_dict(torch.load('save_model/bilstm_crf_best.pth'))

id2tag = {value: key for key, value in conf.tag2id.items()}


def model2test(sample):
    x = []
    for char in sample:
        if char not in word2id:
            char = "UNK"
        x.append(word2id[char])

    x_train = torch.tensor([x])
    mask = (x_train != 0).long()
    model.eval()
    with torch.no_grad():
        if model.name =="BiLSTM":
            outputs = model(x_train, mask)
            preds_ids = torch.argmax(outputs,dim=-1)[0]
            tags = [id2tag[i.item()] for i in preds_ids]
        else:
            preds_ids = model(x_train, mask)
            tags = [id2tag[i] for i in preds_ids[0]]
        chars = [i for i in sample]
        assert len(chars) == len(tags)
        result = extract_entities(chars, tags)
        return result


def extract_entities(tokens, labels):
    entities = []
    entity = []
    entity_type = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):  # 实体的开始
            if entity:  # 如果已经有实体，先保存
                entities.append((entity_type, ''.join(entity)))
                entity = []
            entity_type = label.split('-')[1]
            entity.append(token)
        elif label.startswith("I-") and entity:  # 实体的中间或结尾
            entity.append(token)
        else:
            if entity:  # 保存上一个实体
                entities.append((entity_type, ''.join(entity)))
                entity = []
                entity_type = None

    # 如果最后一个实体没有保存，手动保存
    if entity:
        entities.append((entity_type, ''.join(entity)))

    return {entity: entity_type for entity_type, entity in entities}


if __name__ == '__main__':
    result = model2test(sample='小明的父亲患有冠心病及糖尿病，无手术外伤史及药物过敏史')
    print(result)
