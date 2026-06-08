import torch

from codes.config import Config
from codes.model.casrel_model import CasRel
from codes.utils.process import convert_score_to_zero_one, extract_sub, extract_obj_and_rel

"""
需求：对输入文本进行关系抽取，预测出文本中的三元组（主实体、关系、客实体）。
思路步骤：
1. 实例化CasRel模型并将其移动到指定设备
2. 加载已保存的模型参数
3. 数据处理：
    3.1 使用tokenizer将输入文本转换为id
    3.2 将转换后的结果处理为模型所需的张量形式，并获取句子长度
4. 模型预测：
    4.1 将模型设为评估模式，预测主实体位置信息：
        4.1.1 通过模型对输入进行编码
        4.1.2 基于编码结果预测主实体的头和尾位置
        4.1.3 将预测结果转换为0 - 1形式
        4.1.4 抽取主实体
    4.2 基于主实体预测客实体及关系：
        4.2.1 遍历每个预测出的主实体：
            - 将主实体转成文字，若包含特殊标记[PAD]或[CLS]则跳过
            - 构建主实体相关张量sub_head2tail和sub_len
            - 利用模型预测客实体的头和尾位置
5. 结果解析：
    5.1 处理预测的客实体位置结果，抽取客实体信息及关系
    5.2 遍历每个预测出的客实体，解析关系和客实体文字，若客实体包含特殊标记则跳过
    5.3 组装主实体、关系、客实体为三元组，添加到结果列表
6. 构建并返回结果字典，包含原始文本和预测出的三元组列表
"""

conf = Config()


def model2predict(sample):
    # 1.实例化模型
    model = CasRel(conf).to(conf.device)
    # 2.加载模型参数
    model_path = 'save_model/casrel_model.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 3.处理数据
    text = conf.tokenizer(sample)  # 将句子转成id，默认返回列表
    # print(f'text-->{text}')
    input_ids = torch.tensor([text['input_ids']]).to(conf.device)
    mask = torch.tensor([text['attention_mask']]).to(conf.device)
    # 获取句子长度
    seq_len = len(text['input_ids'])

    # 4.模型预测
    # 4.1 利用模型预测主实体的位置信息
    model.eval()
    with torch.no_grad():
        encoded_text = model.get_encoded_text(input_ids, mask)  # bert编码 [1, 20, 768]
        sub_heads, sub_tails = model.get_subs(encoded_text)  # 预测主实体的位置信息
        pred_sub_heads = convert_score_to_zero_one(sub_heads)  # 转成0-1
        pred_sub_tails = convert_score_to_zero_one(sub_tails)  # 转成0-1
        # 抽取主实体 --> [(head, tail),...]
        pred_subs = extract_sub(pred_sub_heads.squeeze(), pred_sub_tails.squeeze())
        print(f'pred_subs-->{pred_subs}')

        # 4.2 基于主实体信息，预测客实体及关系
        # 用于存储所有三元组
        spo_list = []
        # 模型可能没识别出实体，需要进行判断
        if len(pred_subs) != 0:
            # TODO : pred_subs中可能包含多个实体，所以需要需要循环处理！
            for pred_sub in pred_subs:
                # 1)先将主实体转成文字，然后判断一下该主实体是不是正常的，如果主实体中包含[PAD]或[CLS]，则跳过
                text_list = conf.tokenizer.convert_ids_to_tokens(input_ids[0])  # 获取文本信息列表
                # print(f'text_list-->{text_list}')
                # 通过主实体的索引信息，将主实体信息转成文字
                sub_head_idx = pred_sub[0]  # 主实体的开始索引
                sub_tail_idx = pred_sub[1]  # 主实体的结束索引
                sub = ''.join(text_list[sub_head_idx: sub_tail_idx + 1])  # 获取主实体的值
                # print(f'主实体-->{sub}')
                if '[PAD]' in sub or '[CLS]' in sub:  # 如果主实体中包含[PAD]或[CLS]，则跳过
                    continue

                # 2)构建该实体的 sub_head2tail 和 sub_len
                # 初始化一个全0的主实体张量，然后将开始到结束的索引位置全部赋值为1
                inner_sub_head2tail = torch.zeros(seq_len)
                inner_sub_head2tail[sub_head_idx:sub_tail_idx + 1] = 1
                # print(f'inner_sub_head2tail-->{inner_sub_head2tail.shape}')  # [20]
                # 注意需要将主实体扩展一维，添加batch_size维度
                sub_head2tail = inner_sub_head2tail.unsqueeze(0).to(conf.device)
                # print(f'sub_head2tail-->{sub_head2tail.shape}')  # [1, 20]

                # 主实体长度
                inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
                sub_len = inner_sub_len.unsqueeze(0).to(conf.device)  # 同样需要添加batch_size维度
                # print(f'sub_len-->{sub_len.shape}')  # [1, 1]

                # 3) 预测客实体和关系
                # 因为已经获取了sub_head2tail、sub_len和encoded_text，所以可以直接利用get_objs_for_specific_sub方法预测客实体和关系【推荐】
                pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(encoded_text, sub_head2tail, sub_len)

                # 5.结果解析
                # 抽取客实体信息及关系
                pred_obj_heads = convert_score_to_zero_one(pred_obj_heads)
                pred_obj_tails = convert_score_to_zero_one(pred_obj_tails)
                pred_objs = extract_obj_and_rel(pred_obj_heads[0], pred_obj_tails[0])
                # print(f'pred_objs-->{pred_objs}')  # [(5, 8, 11)]  <-- [(rel_id, obj_head_index, obj_tail_index)]

                if len(pred_objs) == 0:
                    print(f'{pred_sub}没有识别出客实体及关系')
                else:
                    for pred_obj in pred_objs:
                        # 解析客实体信息及关系
                        relation = conf.rel_vocab.to_word(pred_obj[0])  # 把关系id转成关系名称
                        obj_head, obj_tail = pred_obj[1], pred_obj[2]  # 客实体的开始索引和结束索引
                        obj = ''.join(text_list[obj_head: obj_tail + 1])  # 获取客实体的值
                        if '[PAD]' in obj or '[CLS]' in obj:
                            continue
                        # 组装spo三元组
                        sub_spo = {}
                        sub_spo['subject'] = sub
                        sub_spo['predicate'] = relation
                        sub_spo['object'] = obj
                        spo_list.append(sub_spo)

    # 返回结果
    result_dict = {}
    result_dict['text'] = sample
    result_dict['spo_list'] = spo_list
    return result_dict


if __name__ == '__main__':
    sample = "1997年，李柏光从北京大学法律系博士毕业"
    result_dict = model2predict(sample)
    print(result_dict)
