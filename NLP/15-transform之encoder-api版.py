"""
15-transform 之 encoder —— 使用 PyTorch 原生 nn.MultiheadAttention API 版

与手写版 (15-transform之encoder.py) 的对比:
┌─────────────────────────────────────────────────────────────────┐
│  手写版做的事 (30+ 行)          │  API 版 (nn.MultiheadAttention) │
│  ───────────────────────────── │ ─────────────────────────────── │
│  ① clones() 深拷贝4个线性层     │  全部内置，一行 __init__ 搞定    │
│  ② view+transpose 拆分 head    │                                │
│  ③ 调用 attention() 计算点积   │  batch_first=True 直接匹配      │
│  ④ transpose+view 合并 head    │  Embedding 输出格式             │
│  ⑤ 最后一个线性层输出投影       │  CUDA Flash Attention 加速      │
└─────────────────────────────────────────────────────────────────┘

核心 API 参数速查:
  nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
    - embed_dim:   输入输出维度 (如 512)
    - num_heads:   头数 (需能整除 embed_dim，如 8)
    - dropout:     attention 权重上的 dropout
    - batch_first: ★ 设为 True，输入形状 [batch, seq, dim]，不然后面全要 transpose

  forward(query, key, value, key_padding_mask=None, need_weights=True)
    - key_padding_mask: [batch, seq_len] bool，True=忽略该位置(padding)
    - 返回值: (attn_output, attn_weights)
"""

import torch
import torch.nn as nn
import math
from input import *


# ══════════════════════════════════════════════════════════════════════
# 1. 缩放点积注意力函数（底层原理，API 内部已实现，这里保留用于理解）
# ══════════════════════════════════════════════════════════════════════
def attention(query, key, value, mask=None, dropout=None):
    """
    缩放点积注意力计算规则
    公式: Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V

    三步流程:
      1. Q·K^T / √d_k           → 注意力分数 (scores)
      2. softmax(scores)          → 注意力权重 (p_attn)
      3. p_attn · V              → 动态上下文表示 (attn_c)
    """
    d_k = query.size()[-1]                                          # 特征维度

    # Step 1: 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: padding 掩码（将填充位置的分数设为 -∞，softmax 后趋近于 0）
    if mask is not None:
        scores = scores.masked_fill(mask=(mask == 0), value=-1e9)

    # Step 3: softmax → 注意力权重
    p_attn = torch.softmax(scores, dim=-1)

    # Step 4: dropout（训练时随机丢弃部分注意力连接）
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Step 5: 加权求和 → 上下文表示
    attn_c = torch.matmul(p_attn, value)
    return attn_c, p_attn


# ══════════════════════════════════════════════════════════════════════
# 2. 多头注意力机制 —— 直接调用 PyTorch 原生 nn.MultiheadAttention
# ══════════════════════════════════════════════════════════════════════
class MultiHeadedAttention(nn.Module):
    """
    多头注意力 —— 一行 nn.MultiheadAttention 替代 30+ 行手写逻辑

    为什么 API 更好？
      1. 内置 Flash Attention (PyTorch ≥ 2.0): O(n²) 内存 → O(n)，长序列加速 2-5x
      2. 数值稳定: 自动处理 FP16/BF16 混合精度，无需手动调 -1e9 这种 magic number
      3. batch_first=True: 与 Embedding、LSTM 等模块无缝对接，省去 transpose 样板代码
    """

    def __init__(self, head, embedding_dim, dropout_p=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert embedding_dim % head == 0, f'embedding_dim({embedding_dim}) 不能被 head({head}) 整除'

        self.head = head
        self.d_k = embedding_dim // head      # 每个头的维度，如 512÷8=64
        self.attn = None                       # 存储注意力权重，供外部查看/可视化

        # ★★★ 核心: 一行替代所有手写逻辑 ★★★
        # 内部自动完成:
        #   in_proj_weight → 把 Q/K/V 三个线性层合并成一个大矩阵乘法，一次算出
        #   view+transpose → [batch,seq,512] → [batch,8,seq,64]
        #   scaled_dot_product_attention → Flash Attention 加速
        #   transpose+view → [batch,8,seq,64] → [batch,seq,512]
        #   out_proj → 最终线性投影
        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,    # 输入输出维度 512
            num_heads=head,             # 头数 8
            dropout=dropout_p,          # 注意力 dropout
            batch_first=True            # ★ 必须: 匹配 [batch, seq, dim] 格式
        )

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query/key/value: [batch_size, seq_len, embed_dim]
            mask:            旧格式 [batch, 1, 1, seq_len]，uint8，1=有效/0=padding

        返回:
            output: [batch_size, seq_len, embed_dim]
        """
        # ── 掩码格式转换: 旧格式 → key_padding_mask ──
        # 旧: (x!=0).unsqueeze(1).unsqueeze(2) → [batch, 1, 1, seq], 0=pad
        # 新: [batch, seq] bool, True=pad（忽略）
        key_padding_mask = None
        if mask is not None:
            mask_2d = mask.squeeze(1).squeeze(1)       # [batch,1,1,seq] → [batch,seq]
            key_padding_mask = (mask_2d == 0)            # 反转: 0(pad)→True

        # ── 调用 API ──
        output, self.attn = self.mha(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True                             # 保留权重用于分析
        )
        return output


# ══════════════════════════════════════════════════════════════════════
# 3. 前馈全连接层（不变）
# ══════════════════════════════════════════════════════════════════════
class PositionwiseFeedForward(nn.Module):
    """引入非线性变换，增强模型表达能力"""
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)     # 升维 512→2048
        self.linear2 = nn.Linear(d_ff, d_model)     # 降维 2048→512
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
# 4. 测试: 对比手写版与 API 版的输出形状
# ══════════════════════════════════════════════════════════════════════
def dm_test_MultiHeadedAttention():
    vocab = 1000
    d_model = 512
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    # Embedding + PositionalEncoding
    emb = Embeddings(vocab, d_model)
    pe = PositionalEncoding(d_model, 0.1, 60)
    pe_result = pe(emb(x))                              # [2, 4, 512]

    head = 8
    query = key = value = pe_result                     # 自注意力: Q=K=V

    # ── mask 格式说明 ──
    # 旧格式 (兼容手写版接口):
    #   (x != 0) → [2,4] bool
    #   .unsqueeze(1).unsqueeze(2) → [2,1,1,4]  广播到 [2,8,4,4] 的 scores 矩阵
    # 类内部会自动转换为 key_padding_mask: [2,4] bool (True=pad)
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)

    # ── 使用 API 版多头注意力 ──
    my_mha = MultiHeadedAttention(head, d_model, dropout_p=0.1)
    mha_result = my_mha(query, key, value, mask)

    print('=' * 60)
    print('API 版 MultiHeadedAttention 测试')
    print('=' * 60)
    print(f'输入形状:    {pe_result.shape}')             # [2, 4, 512]
    print(f'输出形状:    {mha_result.shape}')             # [2, 4, 512]
    print(f'注意力权重:  {my_mha.attn.shape}')            # [2, 8, 4, 4]
    print(f'参数量:      {sum(p.numel() for p in my_mha.parameters()):,}')
    print()

    # ── 查看 padding 位置的注意力权重是否被正确屏蔽 ──
    # 第2个句子只有前3个token有效，第4个是padding
    # 权重矩阵 [2,8,4,4] 第2句的第4列应全为 ~0
    print('句子2的注意力权重 (head=0, 应看到第4列≈0):')
    print(my_mha.attn[1, 0])                             # [4, 4]


def dm_test_PositionwiseFeedForward():
    """完整测试: Embedding → MultiHeadAttention → FeedForward"""
    vocab = 1000
    d_model = 512
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    emb = Embeddings(vocab, d_model)
    pe = PositionalEncoding(d_model, 0.1, 60)
    pe_result = pe(emb(x))

    head = 8
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)

    # Multi-Head Attention
    my_mha = MultiHeadedAttention(head, d_model, dropout_p=0.1)
    mha_result = my_mha(pe_result, pe_result, pe_result, mask)

    # Feed Forward
    my_pff = PositionwiseFeedForward(d_model=512, d_ff=2048, dropout_p=0.1)
    ff_result = my_pff(mha_result)

    print('=' * 60)
    print('完整 Encoder 子层测试: MHA → FF')
    print('=' * 60)
    print(f'MHA 输出:  {mha_result.shape}')              # [2, 4, 512]
    print(f'FF  输出:  {ff_result.shape}')               # [2, 4, 512]


if __name__ == '__main__':
    dm_test_MultiHeadedAttention()
    print()
    dm_test_PositionwiseFeedForward()
