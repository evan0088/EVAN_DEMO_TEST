# 导入工具包
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from rich import print


# 情感分类任务/文本分类
def dm01_test_text_classification():
    # 1 使用中文预训练模型chinese_sentiment
    # 模型下载地址 git clone https://huggingface.sucp.cn/techthiyanes/chinese_sentiment

    # 2 实例化pipeline对象
    # techthiyanes/xxx 官网自动下载路径
    # 在官网自动下载缓存模型  科学上网
    # my_model = pipeline(task='sentiment-analysis', model='techthiyanes/chinese_sentiment')
    # 本地模型路径
    my_model = pipeline(task='sentiment-analysis', model='../model/chinese_sentiment',device="cuda")
    print('my_model--->', my_model.model)

    # 3 文本送给模型 进行文本分类
    output = my_model('我爱北京天安门，天安门上太阳升。')
    print('output--->', output)
    output = my_model.predict('我爱北京天安门，天安门上太阳升。')
    print('output--->', output)


# 文本特征提取 embedding model
def dm02_test_feature_extraction():
    # 1 下载中文预训练模型 git clone https://huggingface.co/bert-base-chinese

    # 2 实例化pipeline对象 返回模型对象
    my_model = pipeline(task='feature-extraction', model='../model/bert-base-chinese',device="cuda")
    # my_model = pipeline(task='feature-5', model='../model/Qwen3-Embedding-0.6B')
    print('my_model--->', my_model.model)

    # 3 给模型送数据 提取语句特征
    # cls + 文本 + sep
    output = my_model('人生该如何起头')
    print('output--->', type(output), np.array(output).shape)
    # print('output--->', output)


# 完型填空任务 MLM  bert系列的两大预训练任务之一
def dm03_test_fill_mask():
    # 1 下载预训练模型 全词模型git clone https://huggingface.co/hfl/chinese-bert-wwm

    # 2 实例化pipeline对象 返回一个模型
    my_model = pipeline(task='fill-mask', model='../model/chinese-bert-wwm',device="cuda")
    print('my_model--->', my_model.model)

    # 3 给模型送数据 做预测
    input = '我想明天去[MASK]家吃饭。'
    output = my_model(input, top_k=10)

    # 4 输出预测结果
    print('output--->', output)


# 阅读理解任务(抽取式问答) QA
def dm04_test_question_answering():
    # 问答语句
    context = '我叫张三，我是一个程序员，我的喜好是打篮球。'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']

    # 1 下载模型 git clone https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large

    # 2 实例化化pipeline 返回模型
    model = pipeline('question-answering', model='../model/chinese_pretrain_mrc_roberta_wwm_ext_large',device="cuda")
    print('model--->', model.model)

    # 3 给模型送数据 的预测结果
    output = model(context=context, question=questions)
    print('output--->', output)


# 文本摘要任务 总结->长文本进行压缩
def dm05_test_summarization():
    # 1 下载模型 git clone https://huggingface.co/sshleifer/distilbart-cnn-12-6

    # 2 实例化pipline 返回模型
    my_model = pipeline(task='text-generation', model="../model/distilbart-cnn-12-6",device="cuda",max_length=512)

    # 3 准备文本 送给模型
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
    output = my_model(text)

    # 4 打印摘要结果
    print('output--->', output)


# NER任务 命名实体识别
def dm06_test_ner():
    # 1 下载模型 git clone https://huggingface.co/uer/roberta-base-finetuned-cluener2020-chinese

    # 2 实例化pipeline 返回模型
    model = pipeline('ner', model='../model/roberta-base-finetuned-cluener2020-chinese',device="cuda")
    print('model--->', model.model)

    # 3 给模型送数据 打印NER结果
    output = model('我爱北京天安门，天安门上太阳升。鲁迅，五四运动。')
    print('output--->', output)


if __name__ == '__main__':
    # dm01_test_text_classification()
    # dm02_test_feature_extraction()
    # dm03_test_fill_mask()
    # dm04_test_question_answering()
    # dm05_test_summarization()
    dm06_test_ner()