"""
需求：实现一个基于Bert的关系抽取模型CasRel，并对模型进行加载与优化设置。
思路步骤：
    1. **初始化CasRel类**：
        1.1 加载预训练的Bert模型
        1.2 定义四个线性层，分别用于判断主实体头部、主实体尾部、客实体头部及关系类型、客实体尾部及关系类型
    2. **定义获取编码文本的方法**：使用Bert模型对输入的token_ids和mask进行编码，获取编码后的文本
    3. **定义获取主实体预测结果的方法**：通过线性层及sigmoid函数，预测主实体的头部和尾部位置
    4. **定义获取特定主实体对应的客实体预测结果的方法**：
        4.1 将主实体特征与编码后的文本融合
        4.2 对主实体长度扩维并平均主实体信息
        4.3 将处理后的实体特征与原始编码后的文本再次融合
        4.4 通过线性层及sigmoid函数，预测客实体的头部和尾部位置及关系类型
    5. **定义前向传播方法**：
        5.1 获取编码文本
        5.2 获取主实体预测结果
        5.3 获取特定主实体对应的客实体预测结果
        5.4 将预测结果存入字典并返回
    6. **定义计算损失的方法**：
        6.1 根据客实体头部形状确定关系数量，并生成关系掩码
        6.2 分别计算主实体头部、主实体尾部、客实体头部、客实体尾部的损失
        6.3 返回总损失
    7. **定义加载模型的方法**：
        7.1 将模型转移到指定设备
        7.2 对模型参数进行分组，区分需要权重衰减和不需要权重衰减的参数
        7.3 使用AdamW优化器对模型进行优化
        7.4 返回模型、优化器、调度器和设备
    8. **主程序**：初始化配置，调用加载模型的方法
"""

# coding:utf-8
import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import AdamW
from codes.config import *


class CasRel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 定义第一个线性层，来判断主实体的头部位置
        self.sub_heads_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第二个线性层，来判断主实体的尾部位置
        self.sub_tails_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第三个线性层，来判断客实体的头部位置以及关系类型
        self.obj_heads_linear = nn.Linear(conf.bert_dim, conf.num_rel)
        # 定义第四个线性层，来判断客实体的尾部位置以及关系类型
        self.obj_tails_linear = nn.Linear(conf.bert_dim, conf.num_rel)

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        pre_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pre_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pre_sub_heads, pre_sub_tails

    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_text):
        '''
        将subject实体信息融合原始句子中：将主实体字向量实现平均，然后加在当前句子的每一个字向量上，进行计算
        :param sub_head2tail:shape-->【batch_size，1, seq_len】
        :param sub_len:shape--->[batch_size,1]
        :param encoded_text:.shape[batch_size，seq_len，bert_dim]
        :return:
            pred_obj_heads-->shape []
            pre_obj_tails-->shape []
        '''
        sub = torch.matmul(sub_head2tail, encoded_text)  # 将主实体特征和编码后的文本进行融合
        sub_len = sub_len.unsqueeze(1)  # 主实体长度（扩维）
        # 这里为什么要除以sub_len : 因为每个实体的长度不一致 ,如果不除以长度则长实体的向量和会远大于短实体
        sub = sub / sub_len  # 平均主实体信息
        encoded_text = encoded_text + sub  # 将处理后的实体特征和原始编码后的文本进行融合
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pre_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pre_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        '''

        :param input_ids: shape-->[batch_size, seq_len]
        :param mask: shape-->[batch_size, seq_len]
        :param sub_head2tail: shape-->[batch_size, seq_len]
        :param sub_len: shape-->[batch_size, 1]
        :return:
        '''
        # todo: encode_text.shape--->[batch_size,seq_len,bert_dim]
        encoded_text = self.get_encoded_text(input_ids, mask)
        pred_sub_heads, pre_sub_tails = self.get_subs(encoded_text)
        sub_head2tail = sub_head2tail.unsqueeze(1)
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_text)
        result_dict = {'pred_sub_heads': pred_sub_heads,
                       'pred_sub_tails': pre_sub_tails,
                       'pred_obj_heads': pred_obj_heads,
                       'pred_obj_tails': pre_obj_tails,
                       'mask': mask}
        return result_dict

    def compute_loss(self,
                     pred_sub_heads, pred_sub_tails,
                     pred_obj_heads, pred_obj_tails,
                     mask,
                     sub_heads, sub_tails,
                     obj_heads, obj_tails):
        '''
        计算损失
        :param pred_sub_heads:[16, 200, 1]
        :param pred_sub_tails:[16, 200, 1]
        :param pred_obj_heads:[16, 200, 18]
        :param pred_obj_tails:[16, 200, 18]
        :param mask: shape-->[16, 200]
        :param sub_heads: shape-->[16, 200]
        :param sub_tails: shape-->[16, 200]
        :param obj_heads: shape-->[16, 200, 18]
        :param obj_tails: shape-->[16, 200, 18]
        :return:
        '''
        # todo:sub_heads.shape,sub_tails.shape, mask-->[16, 200]
        # todo:obj_heads.shape,obj_tails.shape-->[16, 200, 18]
        rel_count = obj_heads.shape[-1]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)
        loss_1 = self.loss(pred_sub_heads, sub_heads, mask)
        loss_2 = self.loss(pred_sub_tails, sub_tails, mask)
        loss_3 = self.loss(pred_obj_heads, obj_heads, rel_mask)
        loss_4 = self.loss(pred_obj_tails, obj_tails, rel_mask)
        return loss_1 + loss_2 + loss_3 + loss_4

    def loss(self, pred, gold, mask):
        pred = pred.squeeze(-1)
        los = nn.BCELoss(reduction='none')(pred, gold)
        if los.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        los = torch.sum(los * mask) / torch.sum(mask)
        return los


def load_model(conf):
    device = conf.device
    model = CasRel(conf)
    model.to(device)
    # 因为本次模型借助BERT做fine_tuning， 因此需要对模型中的大部分参数进行L2正则处理防止过拟合，包括权重w和偏置b
    # prepare optimzier
    # named_parameters()获取模型中的参数和参数名字
    param_optimizer = list(model.named_parameters())
    print(f'param_optimizer--->{param_optimizer}')
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # no_decay中存放不进行权重衰减的参数{因为bert官方代码对这三项免于正则化}
    # any()函数用于判断给定的可迭代参数iterable是否全部为False，则返回False，如果有一个为True，则返回True
    # 判断param_optimizer中所有的参数。如果不在no_decay中，则进行权重衰减;如果在no_decay中，则不进行权重衰减
    # TODO 不在no_decay里面权重衰减系数=0.01 , 如果在no_decay
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate, eps=10e-8)
    # 是否需要对bert进行warm_up。这里默认不进行
    sheduler = None

    return model, optimizer, sheduler, device


def test_model_forward():
    global model
    conf = Config()
    model = CasRel(conf).to(conf.device)
    import codes.utils.data_loader as data_loader
    train_dataloader, _, _ = data_loader.get_data()
    """
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
        
        results: 
            result_dict = {'pred_sub_heads': pred_sub_heads,
                       'pred_sub_tails': pre_sub_tails,
                       'pred_obj_heads': pred_obj_heads,
                       'pred_obj_tails': pre_obj_tails,
                       'mask': mask}
                       
                       
       def compute_loss(self,
                 pred_sub_heads, pred_sub_tails,
                 pred_obj_heads, pred_obj_tails,
                 mask,
                 sub_heads, sub_tails,
                 obj_heads, obj_tails):
    """
    for inputs, labels in train_dataloader:
        # input_ids, mask, sub_head2tail, sub_len
        # result = model(inputs['input_ids'], inputs['mask'], inputs['sub_head2tail'], inputs['sub_len'])
        """
        **inputs: 相当于把一个dict类型变量， 作为一个key-value的格式的参数传给方法作为入参，相当于
        model(input_ids=inputs['input_ids'],mask=inputs['mask'],sub_head2tail=inputs['sub_head2tail']) )  
        """
        result = model(**inputs)
        for key in result.keys():
            print(f'{key}: {result[key].shape}')

        mask = inputs['mask']
        pred_sub_heads = result['pred_sub_heads']
        pred_sub_tails = result['pred_sub_tails']
        pred_obj_heads = result['pred_obj_heads']
        pred_obj_tails = result['pred_obj_tails']

        sub_heads = labels['sub_heads']
        sub_tails = labels['sub_tails']
        obj_heads = labels['obj_heads']
        obj_tails = labels['obj_tails']

        loss = model.compute_loss(pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails, mask, sub_heads,
                                  sub_tails, obj_heads, obj_tails)
        print(loss)


if __name__ == '__main__':
    # test_model_forward()
    conf = Config()
    device = conf.device
    model = CasRel(conf)
    model.to(device)
    # 因为本次模型借助BERT做fine_tuning， 因此需要对模型中的大部分参数进行L2正则处理防止过拟合，包括权重w和偏置b
    # prepare optimzier
    # named_parameters()获取模型中的参数和参数名字
    for index, (name, param) in enumerate(model.named_parameters()):
        print(f'{index} {name}--->{param.shape}')

