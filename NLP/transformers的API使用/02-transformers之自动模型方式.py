"""
自动模型/具体模型方式:
需要手动调整输入文本以及模型输出, 通过代码进行相应处理
"""

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


# 情感分类任务
def dm01_test_text_classification():
    # 1 自动加载预训练的分词器
    # # techthiyanes/xxx 官网自动下载路径
    # my_tokenizer = AutoTokenizer.from_pretrained('techthiyanes/chinese_sentiment')
    my_tokenizer = AutoTokenizer.from_pretrained('../model/chinese_sentiment')

    # 2 自动加载序列分类模型
    # my_model = AutoModelForSequenceClassification.from_pretrained('techthiyanes/chinese_sentiment')
    my_model = AutoModelForSequenceClassification.from_pretrained('../model/chinese_sentiment')
    print('my_model--->', my_model)
    # 实例化模型配置文件对象

    # 3 文本转张量
    message = '人生该如何起头'

    # 3-1 return_tensors='pt'->pytorch 返回是二维tensor
    # padding=True 填充到批次中最长的序列的长度
    # truncation=True 截断
    # 在encode方法中 补齐和截断不生效 只能接受1条文本/样本
    msg_tensor1 = my_tokenizer.encode(text=message, return_tensors='pt',
                                      padding=True, truncation=True)
    print('msg_tensor1--->', msg_tensor1)

    # 3-2 不用return_tensors='pt'是一维列表
    msg_list2 = my_tokenizer.encode(text=message, padding=True,
                                    truncation=True)
    print('msg_list2--->', msg_list2)
    msg_tensor2 = torch.tensor([msg_list2])
    print('msg_tensor2--->', msg_tensor2)

    # 4 数据送给模型
    # 4-1
    my_model.eval()
    output1 = my_model(msg_tensor2)
    print('情感分类模型头输出outpout1--->', output1)
    # 4-2
    output2 = my_model(msg_tensor2, return_dict=False)
    print('情感分类模型头输出outpout2--->', output2)
    # 4-3
    prob = torch.softmax(output2[0], dim=-1)
    print('概率prob--->', prob)
    class_id = torch.argmax(prob, dim=-1).item()
    print('分类idclass_id--->', class_id)
    # my_model.config: 获取配置信息对象
    # my_model.config.id2label: 获取id对应的标签字典
    print(my_model.config.id2label[class_id])


from transformers import AutoModel


# 特征提取任务
def dm02_test_feature_extraction():
    # 1 加载tokenizer
    my_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='model/bert-base-chinese')

    # 2 加载模型
    my_model = AutoModel.from_pretrained(pretrained_model_name_or_path='model/bert-base-chinese')
    print('my_model--->', my_model)

    # 3 文本转张量
    message = ['你是谁', '人生该如何起头']
    # encode_plus() 的主要功能是将原始文本转换为模型所需的输入格式，包括：
    # 分词（Tokenization）
    # 添加特殊标记（如 [CLS] 和 [SEP]）
    # 转换为 ID 序列（input_ids）
    # 生成注意力掩码（attention_mask）
    # token_type_ids, 当前词属于第1句还是第2句话
    # 填充（padding）或截断（truncation）到指定长度
    # encode_plus: 1条样本 [cls + 文本1 + sep + 文本2 + sep]  问答任务样本形式->[cls + 问题 + sep + 答案 + sep]
    # batch_encode_plus: 多条样本 [[cls + 文本1 + sep + 文本2 + sep], [cls + 文本1 + sep + 文本2 + sep], ...]
    msgs_tensor = my_tokenizer.encode_plus(text=message, return_tensors='pt', truncation=True,
                                           padding='max_length', max_length=30)
    print('msgs_tensor--->', type(msgs_tensor), msgs_tensor)

    # 4 给模型送数据提取特征
    my_model.eval()
    # **msgs_tensor -> 字典拆包  key1=value1 key2=value2
    output = my_model(**msgs_tensor)
    print('output--->', output)
    # last_hidden_state表示最后一个隐藏层的数据
    print('output.last_hidden_state.shape--->', output.last_hidden_state.shape)  # torch.Size([1, 30, 768])
    # pooler_output表示池化，也就是对最后一个隐藏层的cls token再进行线性变换以后平均池化的结果，分类时候使用。
    # cls 存储整个句子的语义表示
    print('output.pooler_output.shape--->', output.pooler_output.shape)  # torch.Size([1, 768])


from transformers import AutoModelForMaskedLM


# 完型填空任务
def dm03_test_fill_mask():
    # 1 加载tokenizer
    modelname = "model/chinese-bert-wwm"
    # modelname = "model/bert-base-chinese"
    my_tokenizer = AutoTokenizer.from_pretrained(modelname)

    # 2 加载模型
    my_model = AutoModelForMaskedLM.from_pretrained(modelname)

    # 3 文本转张量
    input = my_tokenizer.encode_plus('我想明天去[MASK]家吃饭.', return_tensors='pt')
    print('input--->', input)

    # 4 给模型送数据提取特征
    my_model.eval()
    output = my_model(**input)
    print('output--->', output)
    print('output.logits--->', output.logits.shape)  # [1,12,21128]

    # 5 取概率最高
    mask_pred_idx = torch.argmax(output.logits[0][6]).item()
    print('打印概率最高的字索引:', mask_pred_idx)
    print('打印概率最高的字:', my_tokenizer.convert_ids_to_tokens([mask_pred_idx]))


from transformers import AutoModelForSeq2SeqLM


# 文本摘要任务
def dm05_test_summarization():
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
           "sentences that were next to each other in the original text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."

    # 1 加载tokenizer
    my_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="model/distilbart-cnn-12-6")

    # 2 加载模型
    my_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path='model/distilbart-cnn-12-6')
    print('my_model--->', my_model)

    # 3 文本转张量
    # my_tokenizer():本质上调用__call__魔法方法, 最推荐和最灵活的方式，因为它封装了encode_plus和batch_encode_plus的功能
    # 输入可以是单个字符串，也可以是字符串列表(输入是列表会自动进行批处理)
    input = my_tokenizer([text], return_tensors='pt')
    print('input--->', input)

    # 4 送给模型做摘要
    my_model.eval()
    # generate: 直接根据预测概率logits进行词转换, 生成预测词下标
    # output = my_model(input.input_ids)
    # print('output1--->', output)
    output = my_model.generate(input.input_ids)
    print('output2--->', output)

    # 5 处理摘要结果
    # skip_special_tokens:是否去除token前面的特殊字符
    # clean_up_tokenization_spaces:是否清理产生的空格
    summary_text = [my_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for g in output]
    print('summary_text--->', summary_text)
    # convert_ids_to_tokens 函数只能将 ids 还原为 token
    print(my_tokenizer.convert_ids_to_tokens(output[0]))


if __name__ == '__main__':
    dm01_test_text_classification()
    # dm02_test_feature_extraction()
    # dm03_test_fill_mask()
    # dm05_test_summarization()