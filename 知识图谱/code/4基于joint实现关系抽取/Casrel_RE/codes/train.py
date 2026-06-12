"""
需求：实现CasRel模型的训练、验证及相关参数设置与评估指标计算。
思路步骤：
1. 导入所需库及配置、模型、数据加载函数等
2. 定义load_model函数：
    2.1 实例化CasRel模型并将其移动到指定设备
    2.2 对模型参数进行分类，确定哪些参数进行L2正则化（权重衰减）
    2.3 使用AdamW优化器对不同参数组设置不同的权重衰减率和学习率
3. 定义model2train函数进行模型训练：
    3.1 通过get_data函数构建数据迭代器
    3.2 调用load_model函数实例化模型和优化器
    3.3 初始化训练日志相关参数
    3.4 开始训练循环（epoch循环）：
        3.4.1 将模型设为训练模式
        3.4.2 遍历训练数据迭代器：
            - 将数据移至指定设备
            - 模型前向传播得到预测结果
            - 计算损失
            - 梯度清零、反向传播、梯度更新
            - 打印内部训练日志
    3.5 使用验证集评估模型：
        - 将模型设为评估模式
        - 调用model2dev函数得到评估结果
        - 根据评估结果保存最优模型
    3.6 打印外部训练日志
"""
from codes.utils.data_loader import *
from codes.config import *
import pandas as pd
from tqdm import tqdm

from codes.config import Config
from codes.model.casrel_model import load_model
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def model2train(model, train_iter, dev_iter, optimizer, conf):
    epochs = conf.epochs
    best_triple_f1 = 0
    for epoch in range(epochs):
        train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch)
    torch.save(model.state_dict(), 'last_model.pth')


def train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch):
    for step, (inputs, labels) in enumerate(tqdm(train_iter)):
        model.train()
        logist = model(**inputs)
        loss = model.compute_loss(**logist, **labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1500 == 0:
            torch.save(model.state_dict(),
                       '/Casrel_RE/save_model/epoch_%s_model_%s.pth' % (epoch, step))
            results = model2dev(model, dev_iter)
            print(results[-1])
            if results[-2] > best_triple_f1:
                best_triple_f1 = results[-2]
                torch.save(model.state_dict(), '../save_model/best_f1.pth')
                print('epoch:{},'
                      'step:{},'
                      'sub_precision:{:.4f}, '
                      'sub_recall:{:.4f}, '
                      'sub_f1:{:.4f}, '
                      'triple_precision:{:.4f}, '
                      'triple_recall:{:.4f}, '
                      'triple_f1:{:.4f},'
                      'train loss:{:.4f}'.format(epoch,
                                                 step,
                                                 results[0],
                                                 results[1],
                                                 results[2],
                                                 results[3],
                                                 results[4],
                                                 results[5],
                                                 loss.item()))

    return best_triple_f1


def model2dev(model, dev_iter):
    '''
    验证模型效果
    :param model:
    :param dev_iter:
    :return:
    '''
    model.eval()
    # 定义一个df，来展示模型的指标。
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    for inputs, labels in tqdm(dev_iter):
        logist = model(**inputs)
        pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        batch_size = inputs['input_ids'].shape[0]
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

        for batch_index in range(batch_size):

            pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(),
                                    pred_sub_tails[batch_index].squeeze())

            true_subs = extract_sub(sub_heads[batch_index].squeeze(),
                                    sub_tails[batch_index].squeeze())

            pred_ojbs = extract_obj_and_rel(pred_obj_heads[batch_index],
                                            pred_obj_tails[batch_index])

            true_objs = extract_obj_and_rel(obj_heads[batch_index],
                                            obj_tails[batch_index])

            df['PRED']['sub'] += len(pred_subs)
            df['REAL']['sub'] += len(true_subs)

            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df['TP']['sub'] += 1

            df['PRED']['triple'] += len(pred_ojbs)
            df['REAL']['triple'] += len(true_objs)
            for true_obj in true_objs:
                if true_obj in pred_ojbs:
                    df['TP']['triple'] += 1

    df.loc['sub', 'p'] = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    df.loc['sub', 'r'] = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    df.loc['sub', 'f1'] = 2 * df['p']['sub'] * df['r']['sub'] / (df['p']['sub'] +
                                                                 df['r']['sub'] +
                                                                 1e-9)
    sub_precision = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    sub_recall = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    df.loc['triple', 'p'] = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    df.loc['triple', 'r'] = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    df.loc['triple', 'f1'] = 2 * df['p']['triple'] * df['r']['triple'] / (
            df['p']['triple'] + df['r']['triple'] + 1e-9)

    triple_precision = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    triple_recall = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (
            triple_precision + triple_recall + 1e-9)

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


if __name__ == '__main__':
    conf = Config()

    model, optimizer, _, device = load_model(conf)

    train_data, dev_data, _ = get_data()
    model2train(model, train_data, dev_data, optimizer, conf)
