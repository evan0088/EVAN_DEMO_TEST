"""
需求：实现数据处理相关功能，包括查找实体索引位置、创建标签以及整理数据批次。
思路步骤：
1. 准备：导入必要的模块和配置信息。
2. 定义`find_head_idx`函数，用于获取实体在源文本中的开始索引位置：
    2.1 遍历源文本，检查是否存在与目标文本相同的子串，若存在则返回起始索引，否则返回 -1。
3. 定义`create_label`函数，用于获取每个样本的主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示：
    3.1 初始化各类张量，包括主实体开始和结束位置、客实体开始和结束位置、主实体从开头到末尾词的索引、主实体长度等。
    3.2 处理数据中可能存在的无关系三元组情况，初始化主词长度为1以防止零除报错。
    3.3 初始化主词到谓词的映射字典`s2ro_map`。
    3.4 遍历每个三元组，将其转换为所需格式，并查找主实体和客实体的开始索引。
    3.5 若主实体和客实体的开始索引都有效，则更新`s2ro_map`。
    3.6 若`s2ro_map`不为空，根据其中的信息更新各类张量。
4. 定义`collate_fn`函数，用于整理数据批次：
    4.1 从输入数据中提取文本列表和三元组列表。
    4.2 使用tokenizer对文本列表进行编码并补齐。
    4.3 循环遍历每个样本，调用`create_label`函数获取各类实体信息并保存到相应列表中。
    4.4 将各类数据转换为张量并移动到指定设备。
    4.5 整理输入数据和标签数据并返回。
"""
# coding:utf-8
from codes.config import *
import torch
from random import choice
from collections import defaultdict

# 配置对象实例化
conf = Config()


# 获取实体的开始索引位置
def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


# 获取每个样本的：主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
def create_label(inner_triples, inner_input_ids, seq_len):
    inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
    inner_obj_heads = torch.zeros((seq_len, conf.num_rel))
    inner_obj_tails = torch.zeros((seq_len, conf.num_rel))
    inner_sub_head2tail = torch.zeros(seq_len)  # 随机抽取一个实体，从开头一个词到末尾词的索引

    # 因为数据预处理代码还待优化,会有不存在关系三元组的情况，
    # 初始化一个主词的长度为1，即没有主词默认主词长度为1，
    # 防止零除报错,初始化任何非零数字都可以，没有主词分子是全零矩阵
    inner_sub_len = torch.tensor([1], dtype=torch.float)
    # 主词到谓词的映射
    s2ro_map = defaultdict(list)
    # print(s2ro_map)
    for inner_triple in inner_triples:
        # print(inner_triple)
        inner_triple = (
            conf.tokenizer(inner_triple['subject'], add_special_tokens=False)['input_ids'],
            conf.rel_vocab.to_index(inner_triple['predicate']),
            conf.tokenizer(inner_triple['object'], add_special_tokens=False)['input_ids']
        )
        sub_head_idx = find_head_idx(inner_input_ids, inner_triple[0])
        obj_head_idx = find_head_idx(inner_input_ids, inner_triple[2])
        if sub_head_idx != -1 and obj_head_idx != -1:
            sub = (sub_head_idx, sub_head_idx + len(inner_triple[0]) - 1)
            # s2ro_map保存主语到谓语的映射
            s2ro_map[sub].append(
                (obj_head_idx, obj_head_idx + len(inner_triple[2]) - 1, inner_triple[1]))  # {(3,5):[(7,8,0)]} 0是关系
    if s2ro_map:
        for s in s2ro_map:
            inner_sub_heads[s[0]] = 1
            inner_sub_tails[s[1]] = 1
        sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
        inner_sub_head2tail[sub_head_idx:sub_tail_idx + 1] = 1
        inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
        for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
            inner_obj_heads[ro[0]][ro[2]] = 1
            inner_obj_tails[ro[1]][ro[2]] = 1
    return inner_sub_len, inner_sub_head2tail, inner_sub_heads, inner_sub_tails, inner_obj_heads, inner_obj_tails


def collate_fn(data):
    text_list = [value[0] for value in data]
    triple = [value[1] for value in data]
    # 按照batch中最长句子补齐
    text = conf.tokenizer.batch_encode_plus(text_list, padding=True)
    batch_size = len(text['input_ids'])
    seq_len = len(text['input_ids'][0])
    sub_heads = []
    sub_tails = []
    obj_heads = []
    obj_tails = []
    sub_len = []
    sub_head2tail = []
    # 循环遍历每个样本，将实体信息进行张量的转化
    for batch_index in range(batch_size):
        inner_input_ids = text['input_ids'][batch_index]  # 单个句子变成索引后
        inner_triples = triple[batch_index]
        # 获取每个样本的：主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
        results = create_label(inner_triples, inner_input_ids, seq_len)
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])
    input_ids = torch.tensor(text['input_ids']).to(conf.device)
    mask = torch.tensor(text['attention_mask']).to(conf.device)
    # 借助torch.stack()函数沿一个新维度对输入batch_size张量序列进行连接，序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度,
    sub_heads = torch.stack(sub_heads).to(conf.device)
    sub_tails = torch.stack(sub_tails).to(conf.device)
    sub_len = torch.stack(sub_len).to(conf.device)
    sub_head2tail = torch.stack(sub_head2tail).to(conf.device)
    obj_heads = torch.stack(obj_heads).to(conf.device)
    obj_tails = torch.stack(obj_tails).to(conf.device)

    inputs = {
        'input_ids': input_ids,
        'mask': mask,
        'sub_head2tail': sub_head2tail,
        'sub_len': sub_len
    }
    labels = {
        'sub_heads': sub_heads,
        'sub_tails': sub_tails,
        'obj_heads': obj_heads,
        'obj_tails': obj_tails
    }

    return inputs, labels




def extract_sub(pred_sub_heads, pred_sub_tails):
    '''
    :param pred_sub_heads: 模型预测出的主实体开头位置
    :param pred_sub_tails: 模型预测出的主实体尾部位置
    :return: subs列表里面对应的所有实体【head, tail】
    '''
    subs = []
    # 统计预测出所有值为1的元素索引位置
    heads = torch.arange(0, len(pred_sub_heads), device=conf.device)[pred_sub_heads == 1]
    tails = torch.arange(0, len(pred_sub_tails), device=conf.device)[pred_sub_tails == 1]
    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))
    return subs


def extract_obj_and_rel(obj_heads, obj_tails):
    '''

    :param obj_heads:  模型预测出的从实体开头位置以及关系类型
    :param obj_tails:  模型预测出的从实体尾部位置以及关系类型
    :return: obj_and_rels：元素形状：(rel_index, start_index, end_index)
    '''
    obj_heads = obj_heads.T
    obj_tails = obj_tails.T
    rel_count = obj_heads.shape[0]
    obj_and_rels = []

    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]
        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_index, end_index = obj
                obj_and_rels.append((rel_index, start_index, end_index))
    return obj_and_rels


def convert_score_to_zero_one(tensor):
    '''
    以0.5为阈值，大于0.5的设置为1，小于0.5的设置为0
    '''
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor