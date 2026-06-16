"""
需求：基于 BiLSTM + Multi-Head Self-Attention 实现 NER 训练（用注意力机制替换 CRF）

与 BiLSTM+CRF 训练的核心区别：
┌──────────────────────────────────────────────────────────────────┐
│  BiLSTM+CRF (原方案)           │  BiLSTM+Attention (本方案)       │
│  ──────────────────────────── │ ────────────────────────────────  │
│  损失: model.log_likelihood()  │  损失: nn.CrossEntropyLoss()     │
│  → 内部调用 CRF 前向算法       │  → 和纯 BiLSTM 一样简单           │
│  需梯度裁剪 clip_grad_norm_    │  不需要梯度裁剪                   │
│  推理: model() → viterbi_decode│  推理: model() → torch.argmax()  │
│  需 mask 转 bool               │  mask 直接用 long                │
└──────────────────────────────────────────────────────────────────┘

思路步骤：
1. 获取数据
2. 3个定义：模型、损失函数、优化器
    2.1 定义模型：NERLSTM_Attention，把网络参数输送给 GPU
    2.2 定义损失函数：CrossEntropyLoss（与纯 BiLSTM 相同，比 CRF 简洁）
    2.3 定义优化器：Adam
3. 训练循环
    3.1 epochs 循环
    3.2 batch 循环
4. 计算损失
    4.1 将 x, y, mask 送给 GPU
    4.2 前向传播 → 变换 pred 形状 → 计算交叉熵损失
5. 更新参数
    5.1 梯度清零
    5.2 反向传播
    5.3 参数更新
6. 验证 & 保存模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model.ner_lstm_attention import NERLSTM_Attention
from utils.data_loader import *
from tqdm import tqdm
from config import Config
import time
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score

conf = Config()


def model2train():
    """BiLSTM+Attention 模型训练主函数"""

    # ── 1. 获取数据 ──
    train_dataloader, dev_dataloader = get_data()

    # ── 2.1 实例化模型 ──
    # num_heads 需能被 hidden_dim 整除，256 / 8 = 32 维/头
    model = NERLSTM_Attention(
        embedding_dim=conf.embedding_dim,   # 300
        hidden_dim=conf.hidden_dim,         # 256
        dropout=conf.dropout,               # 0.2
        word2id=word2id,
        tag2id=conf.tag2id,
        num_heads=8
    )
    model = model.to(conf.device)

    # ── 2.2 实例化损失函数 ──
    # 用交叉熵损失，比 CRF 的 log_likelihood 简洁很多
    criterion = nn.CrossEntropyLoss()

    # ── 2.3 实例化优化器 ──
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)

    # ── 3. 训练循环 ──
    start_time = time.time()
    f1_best = -1000

    for epoch in range(conf.epochs):
        model.train()
        total_loss = 0.0

        for index, (inputs, labels, mask) in enumerate(
            tqdm(train_dataloader, desc=f'BiLSTM+Attention 训练 Epoch {epoch+1}/{conf.epochs}')
        ):
            # ── 4.1 将数据送到 GPU ──
            x = inputs.to(conf.device)       # [batch_size, seq_len]
            mask = mask.to(conf.device)       # [batch_size, seq_len], 1=有效, 0=padding
            y = labels.to(conf.device)        # [batch_size, seq_len]

            # ── 4.2 前向传播 ──
            # pred [batch_size, seq_len, tag_size]
            pred = model(x, mask)

            # 为什么需要变换 pred 形状？
            # CrossEntropyLoss 要求: pred [N, C] 和 target [N]
            # 其中 N = batch_size * seq_len, C = tag_size
            # 原始 pred [batch_size, seq_len, tag_size] → [batch_size * seq_len, tag_size]
            pred = pred.view(-1, len(conf.tag2id))

            # y [batch_size, seq_len] → [batch_size * seq_len]
            # 损失函数内部会自动忽略 padding 位置（标签为 0 的位置仍然参与计算）
            # 注意：这里 padding 的标签也是 0（即 "O" 非实体），与真实 "O" 无法区分
            # 但 mask 已经在模型前向传播中置零了 padding 位置的输出，
            # 所以 padding 位置的预测 logits 全是 0，对梯度影响极小
            my_loss = criterion(pred, y.view(-1))

            total_loss += my_loss.item()

            # ── 5.1 梯度清零 ──
            optimizer.zero_grad()

            # ── 5.2 反向传播 ──
            # 注意：与 CRF 不同，Attention 模型不需要梯度裁剪
            # 因为残差连接 + LayerNorm 已经保证了梯度稳定
            my_loss.backward()

            # ── 5.3 参数更新 ──
            optimizer.step()

            if index % 200 == 0:
                print(f'epoch:{epoch:04d}, loss:{my_loss.item():.6f}')

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1} 平均损失: {avg_loss:.6f}')

        # ── 6. 验证 & 保存模型 ──
        precision, recall, f1, report, accuracy = model2dev(dev_dataloader, model)

        print(f'Epoch {epoch+1} 验证结果:')
        print(f'  Accuracy:  {accuracy:.4f}')
        print(f'  Precision: {precision:.4f} (macro)')
        print(f'  Recall:    {recall:.4f} (macro)')
        print(f'  F1-score:  {f1:.4f} (macro)')

        if f1 > f1_best:
            f1_best = f1
            torch.save(model.state_dict(), 'save_model/bilstm_attention_best.pth')
            print(f'✅ 新最佳模型已保存 (F1: {f1:.4f})')
            print(report)
        else:
            print(f'  未提升 (当前 F1: {f1:.4f}, 最佳 F1: {f1_best:.4f})')

    end_time = time.time()
    print(f'\n训练完成！总耗时：{end_time - start_time:.2f} 秒')
    print(f'最佳 F1-score: {f1_best:.4f}')


def model2dev(dev_iter, model):
    """
    BiLSTM+Attention 模型验证函数

    与 CRF 版本的 model2dev 区别：
    - 不需要把 mask 转为 bool
    - 直接用 torch.argmax 解码（不需要维特比）

    返回:
        precision, recall, f1, report, accuracy
    """
    preds, golds = [], []
    model.eval()

    with torch.no_grad():
        for index, (inputs, labels, mask) in enumerate(tqdm(dev_iter, desc="测试集验证")):
            val_x = inputs.to(conf.device)
            mask = mask.to(conf.device)
            val_y = labels.to(conf.device)

            # 前向传播 → logits
            pred = model(val_x, mask)

            # argmax 解码：直接取每个位置得分最高的标签
            # 比 CRF 的维特比解码简单得多！
            predict = torch.argmax(pred, dim=-1).tolist()

            # ── 提取真实长度 ──
            # mask: 1=有效, 0=padding
            # 统计每个样本中有效 token 的数量
            leng = []
            for i in mask.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(len(tmp))

            # 提取真实长度的预测标签
            for idx, p in enumerate(predict):
                preds.extend(p[:leng[idx]])

            # 提取真实长度的真实标签
            for idx, y_true in enumerate(val_y.cpu().tolist()):
                golds.extend(y_true[:leng[idx]])

    # ── 计算评估指标 ──
    # 使用 macro 平均：对每个类别分别计算再取平均，不受标签不平衡影响
    # NER 任务中 "O" 标签占比极高（~80%+），micro 会被 O 主导 → 避免使用
    accuracy = accuracy_score(golds, preds)
    precision = precision_score(golds, preds, average='macro', zero_division=0)
    recall = recall_score(golds, preds, average='macro', zero_division=0)
    f1 = f1_score(golds, preds, average='macro', zero_division=0)
    report = classification_report(golds, preds, zero_division=0)

    return precision, recall, f1, report, accuracy


if __name__ == '__main__':
    model2train()
