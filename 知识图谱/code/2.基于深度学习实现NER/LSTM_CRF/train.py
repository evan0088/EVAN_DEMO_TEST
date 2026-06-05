"""
需求：在一个函数中实现BiLSTM和BiLSTM-CRF模型的训练

思路步骤：
注意：下面每个步骤都需要考虑两种不同模型
1. 获取数据
2.基于深度学习实现NER. 3个定义：模型、损失函数、优化器
    2.基于深度学习实现NER.1 定义模型：NerBiLSTM 、NerBiLSTMCrf，把网络参数输送给GPU
    2.基于深度学习实现NER.2.基于深度学习实现NER 定义损失函数：由于NerBiLSTMCrf使用自定义loss计算，这里只需要定义一个交叉熵损失
    2.基于深度学习实现NER.3 定义优化器：使用adam
3. 2个循环
    3.1 epochs
    3.2.基于深度学习实现NER batch
4. 计算损失
    4.1 将x, y ,mask送给GPU
    4.2.基于深度学习实现NER 计算损失
        ① 对于NerBiLSTM： 第一步、前向传播 第二步、计算交叉熵损失
        ② 对于NerBiLSTMCrf：使用log_likelihood()
5. 更新参数
    5.1 梯度清零
    5.2.基于深度学习实现NER 反向传播：对于模型进行反向传播。 对于bi-lstm+crf还需要做梯度裁剪
    5.3 参数更新
6. 保存模型
    6.1 评估模型
    6.2.基于深度学习实现NER 保存模型网络参数

"""

import torch
import torch.nn as nn
import torch.optim as optim
from model.ner_lstm import *
from model.ner_lstm_crf import *
from utils.data_loader import *
from tqdm import tqdm
# classification_report可以导出字典格式，修改参数：output_dict=True，可以将字典在保存为csv格式输出
from config import *
import time
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

conf = Config()


def model2train():
    # 获取数据
    train_dataloader, dev_dataloader = get_data()
    # 1. 实例化模型
    models = {'BiLSTM': NERLSTM,
              'BiLSTM_CRF': NERLSTM_CRF}
    #                                300               256              0.2         len(7836)         len(11)
    model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
    model = model.to(conf.device)
    # 2.基于深度学习实现NER. 实例化损失函数
    criterion = nn.CrossEntropyLoss()
    # 3. 实例化优化器
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    # 选择模型进行训练
    start_time = time.time()
    if conf.model == 'BiLSTM':
        f1_score = -1000
        for epoch in range(conf.epochs):
            model.train()
            for index, (inputs, labels, mask) in enumerate(tqdm(train_dataloader, desc='BiLSTM训练')):
                x = inputs.to(conf.device)
                mask = mask.to(conf.device)
                y = labels.to(conf.device)
                # pred [batch_size, seq_len, tag_size]
                pred = model(x, mask)
                """
                
                Q1. 为什么在Bilstm计算完还需要变换pred的形状,为什么要降维为二维矩阵
                A1:  因为交叉熵损失函数只能接受两个参数: pred(二维) ,y_true(一维)
                因为这里是NER任务,采用BIO标注,所以每一个词都标有标签所以
                pred[batch_size, seq_len, tag_size] 分别是批次大小,句子长度(已统一),tag_size 每个词的倾向度
                y_true [batch_size , seq_len] 每个词的真实标签
                """
                # pred[batch_size, seq_len, tag_size] ---> pred [batch_size * seq_len, tag_size]
                pred = pred.view(-1, len(conf.tag2id))
                # pred [batch_size * seq_len, tag_size] y真实标签值 [batch_size * seq_len]
                my_loss = criterion(pred, y.view(-1))
                # 梯度清零
                optimizer.zero_grad()
                # 反向传播
                my_loss.backward()
                # 更新参数
                optimizer.step()
                if index % 200 == 0:
                    print('epoch:%04d,------------loss:%f' % (epoch, my_loss.item()))
            precision, recall, f1, report = model2dev(dev_dataloader, model)
            if f1 > f1_score:
                f1_score = f1
                torch.save(model.state_dict(), 'save_model/bilstm_best.pth')
                print(report)
        end_time = time.time()
        print(f'训练总耗时：{end_time - start_time}')
    elif conf.model == 'BiLSTM_CRF':
        f1_score = -1000
        for epoch in range(conf.epochs):
            model.train()
            for index, (inputs, labels, mask) in enumerate(tqdm(train_dataloader, desc='bilstm+crf训练')):
                x = inputs.to(conf.device)
                mask = mask.to(torch.bool).to(conf.device)
                tags = labels.to(conf.device)
                # CRF
                loss = model.log_likelihood(x, tags, mask).mean()
                optimizer.zero_grad()
                loss.backward()
                # CRF
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
                optimizer.step()
                if index % 200 == 0:
                    print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))
            precision, recall, f1, report = model2dev(dev_dataloader, model)
            if f1 > f1_score:
                f1_score = f1
                torch.save(model.state_dict(), 'save_model/bilstm_best.pth')
                print(report)
        end_time = time.time()
        print(f'训练总耗时：{end_time - start_time}')


"""
需求：在一个函数中实现BiLSTM和BiLSTM-CRF模型的验证
思路步骤：
1. 获取数据
2.基于深度学习实现NER. 开启评估模式
3. 前向传播，得到预测值
    3.1 把x, tags, mask提交给GPU
    3.2.基于深度学习实现NER 执行前向传播
4. 计算评估指标
    4.1 计算精准率
    4.2.基于深度学习实现NER 计算召回率
    4.3 计算f1-score
    4.4 获取分类结果报告
"""


def model2dev(dev_iter, model):
    preds, golds = [], []
    model.eval()
    for index, (inputs, labels, mask) in enumerate(tqdm(dev_iter, desc="测试集验证")):
        val_x = inputs.to(conf.device)
        mask = mask.to(conf.device)
        val_y = labels.to(conf.device)
        predict = []
        if model.name == "BiLSTM":
            pred = model(val_x, mask)
            predict = torch.argmax(pred, dim=-1).tolist()
        elif model.name == "BiLSTM_CRF":
            mask = mask.to(torch.bool)
            predict = model(val_x, mask)
        # 统计非0的，也就是真实标签的长度
        # 4. TODO 计算评估指标 ： 获取真实长度预测值和真实值
        leng = []
        # TODO val_y因为我们进行padding的时候补的是0，非实体O也是0，在这里我们无法判断它是补充的还是本身就是非实体
        for i in mask.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j.item())
            leng.append(len(tmp))
        # 提取真实长度的预测标签

        for index, i in enumerate(predict):
            preds.extend(i[:leng[index]])

        # 提取真实长度的真实标签
        for index, i in enumerate(val_y.cpu().tolist()):
            golds.extend(i[:leng[index]])
    """
    TODO average的不同策略
    ``'binary'``:
        仅报告由 ``pos_label`` 指定的类别的结果。
        这仅适用于目标（``y_{true,pred}``）是二分类的情况。
    ``'micro'``:
        通过计算总真正例、假反例和假正例来全局计算指标。
    ``'macro'``:
        计算每个标签的指标，并求其未加权的平均值。
        这不会考虑标签不平衡。
    ``'weighted'``:
        计算每个标签的指标，并找到其按支持度（每个标签的真实实例数）加权的平均值。
        这会调整 'macro' 以考虑标签不平衡；可能导致 F-score 不在精确度和召回率之间。
    ``'samples'``:
        计算每个实例的指标，并求其平均值（仅适用于多标签分类，且此处与 :func:`accuracy_score` 不同的情况）。
    """
    precision = precision_score(golds, preds, average='macro')
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    report = classification_report(golds, preds)

    return precision, recall, f1, report


if __name__ == '__main__':
    model2train()
