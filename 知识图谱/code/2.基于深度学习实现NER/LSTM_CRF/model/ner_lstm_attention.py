"""
需求：定义 BiLSTM + Multi-Head Self-Attention 模型类（用注意力机制替换 CRF）

与 BiLSTM+CRF 的对比：
┌─────────────────────────────────────────────────────────────────┐
│  BiLSTM+CRF                    │  BiLSTM+Attention (本模型)      │
│  ───────────────────────────── │ ──────────────────────────────── │
│  Embedding                     │  Embedding                      │
│  BiLSTM                        │  BiLSTM                         │
│  Linear → emission scores      │  Multi-Head Self-Attention      │
│  CRF (转移矩阵 + 维特比解码)    │  Residual + LayerNorm           │
│  log_likelihood() 损失         │  Linear → tag logits            │
│                                │  CrossEntropyLoss (简洁!)       │
└─────────────────────────────────────────────────────────────────┘

优势：
  1. 训练更简单：直接用交叉熵损失，无需 CRF 的 log_likelihood 计算
  2. 推理更快：argmax 直接解码，无需维特比算法
  3. 全局依赖：Self-Attention 让每个位置关注全句所有位置，隐式学习标签依赖
  4. 梯度稳定：残差连接 + LayerNorm，不会像 CRF 那样出现梯度爆炸

思路步骤：
1. 定义模型类：接收 embedding_dim, hidden_dim, dropout, word2id, tag2id, num_heads
    1.1 继承 nn.Module，初始化父类
    1.2 定义模型的超参数
    1.3 搭建神经网络：
        ① Embedding 层
        ② BiLSTM 层
        ③ Multi-Head Self-Attention 层（替换 CRF，捕捉全局上下文依赖）
        ④ LayerNorm（稳定训练）
        ⑤ Dropout
        ⑥ 隐层转 tag 线性层

2. 实现模型的前向传播方法：接收 x, mask
    2.1 词嵌入
    2.2 BiLSTM 前向传播
    2.3 Multi-Head Self-Attention（带 padding mask）
    2.4 残差连接 + LayerNorm
    2.5 Dropout + 掩码计算
    2.6 线性变换 → tag logits
"""

import torch
import torch.nn as nn
import config
import utils.common as common
from utils.data_loader import get_data


class NERLSTM_Attention(nn.Module):
    """
    BiLSTM + Multi-Head Self-Attention 用于 NER 序列标注

    核心思路：
    - BiLSTM 提取序列的上下文特征（局部依赖）
    - Multi-Head Self-Attention 让每个位置关注全句所有位置（全局依赖）
    - 残差连接 + LayerNorm 保证梯度流畅，训练稳定
    - 最后线性层映射到 tag 空间，用 CrossEntropyLoss 训练
    """

    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id, num_heads=8):
        """
        参数:
            embedding_dim: 词向量维度 (如 300)
            hidden_dim: BiLSTM 隐层维度 (如 256)，也是 Attention 的输入维度
            dropout: dropout 比率
            word2id: 词 → id 映射表
            tag2id: 标签 → id 映射表
            num_heads: 多头注意力的头数（默认 8，需能被 hidden_dim 整除）
        """
        super(NERLSTM_Attention, self).__init__()

        # ── 1.2 定义模型的超参数 ──
        self.name = "BiLSTM_Attention"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1       # +1 为 PAD (index=0)
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)
        self.num_heads = num_heads

        # ── 1.3 搭建神经网络 ──

        # ① Embedding 层：将 token id 映射为稠密词向量
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

        # ② BiLSTM 层：双向 LSTM 提取序列上下文
        #    hidden_dim // 2 是因为双向拼接后总维度 = hidden_dim
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # ③ Multi-Head Self-Attention 层（替换 CRF 的核心模块）
        #    让每个 token 关注句子中所有其他 token，捕捉全局依赖
        #    效果类似于 CRF 学习标签转移概率，但更灵活：
        #    - 不同 head 关注不同方面（语法、语义、标签相容性）
        #    - 隐式学习 BIO 约束（如 I-XXX 不应出现在 B-XXX 之前）
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,   # 输入/输出维度 = BiLSTM 输出维度
            num_heads=num_heads,          # 多头数
            dropout=dropout,              # 注意力内部的 dropout
            batch_first=True              # 输入 shape: (batch, seq_len, hidden_dim)
        )

        # ④ LayerNorm：对 Attention 输出做归一化，稳定训练
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # ⑤ Dropout：防止过拟合
        self.dropout = nn.Dropout(dropout)

        # ⑥ 线性层：hidden_dim → tag_size，得到每个标签的 logits
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

    def forward(self, x, mask):
        """
        前向传播

        参数:
            x:    [batch_size, seq_len]          输入 token id
            mask: [batch_size, seq_len]          注意力掩码 (1=有效, 0=padding)

        返回:
            outputs: [batch_size, seq_len, tag_size]  每个位置每个标签的 logits
        """
        # ── 2.1 词嵌入 ──
        # x [batch_size, seq_len] → embedding [batch_size, seq_len, embedding_dim]
        embedding = self.word_embeds(x)

        # ── 2.2 BiLSTM 前向传播 ──
        # embedding [batch_size, seq_len, embedding_dim]
        #   → lstm_out [batch_size, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(embedding)

        # ── 2.3 Multi-Head Self-Attention ──
        # Q=K=V=lstm_out，让每个位置对全句做 self-attention
        #
        # key_padding_mask: True 表示"忽略该位置"
        #   原始 mask:  1=有效, 0=padding
        #   转换后:     False=有效, True=padding（符合 PyTorch API）
        attn_padding_mask = (mask == 0)

        # attn_out [batch_size, seq_len, hidden_dim]
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=attn_padding_mask
        )

        # ── 2.4 残差连接 + LayerNorm ──
        # 残差连接：保留 BiLSTM 的原始信息，让 Attention 学习"增量"特征
        # LayerNorm：归一化到稳定分布，加速收敛
        attn_out = self.layer_norm(lstm_out + attn_out)

        # ── 2.5 Dropout + 掩码计算 ──
        attn_out = self.dropout(attn_out)

        # 将 padding 位置的输出置零，避免影响后续计算
        # mask.unsqueeze(-1): [batch_size, seq_len, 1]
        attn_out = attn_out * mask.unsqueeze(-1)

        # ── 2.6 线性变换 → tag logits ──
        # attn_out [batch_size, seq_len, hidden_dim]
        #   → outputs [batch_size, seq_len, tag_size]
        outputs = self.hidden2tag(attn_out)

        return outputs


if __name__ == '__main__':
    # 快速测试：打印模型结构和输出形状
    datas, word2id = common.build_data()
    conf = config.Config()

    model = NERLSTM_Attention(
        embedding_dim=conf.embedding_dim,
        hidden_dim=conf.hidden_dim,
        dropout=conf.dropout,
        word2id=word2id,
        tag2id=conf.tag2id,
        num_heads=8                # 256 / 8 = 32 维/头
    )

    train_data_loader, dev_data_loader = get_data()

    for input_ids_padded, labels_padded, attention_mask in dev_data_loader:
        outputs = model(input_ids_padded, attention_mask)
        print(f'模型名称: {model.name}')
        print(f'input_ids_padded:  {input_ids_padded.shape}')
        print(f'attention_mask:    {attention_mask.shape}')
        print(f'outputs (logits):  {outputs.shape}')
        print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
        print('*' * 100)
        break
