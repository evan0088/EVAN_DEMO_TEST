import torch
import torch.nn as nn
from config import Config
from utils import build_dataloader

conf = Config()

class BiLSTMClassifier(nn.Module):
    """
    简化的BiLSTM分类模型，用于文本分类任务。
    通过嵌入层、双向LSTM层、最大池化、Dropout和全连接层处理输入序列，输出分类logits。
    """
    def __init__(self, config):
        """
        初始化BiLSTMClassifier模型。

        Args:
            config: 配置对象，包含模型超参数，如vocab_size, embed_size, hidden_size_lstm等。
        """
        super(BiLSTMClassifier, self).__init__()
        # Step 1: 初始化嵌入层，将token ID映射为嵌入向量
        # 变量名: self.embedding
        # 核心函数: nn.Embedding
        # 输入: token ID [batch_size, seq_len]
        # 输出: 嵌入向量 [batch_size, seq_len, embed_size]
        self.embedding = nn.Embedding(
            num_embeddings=config.tokenizer.vocab_size,  # 词汇表大小
            embedding_dim=config.embed_size  # 嵌入维度
        )

        # Step 2: 初始化双向LSTM层，提取序列特征
        # 变量名: self.lstm
        # 核心函数: nn.LSTM #embed_size hidden_size_lstm num_layers
        self.lstm = nn.LSTM(
            input_size=config.embed_size,  # 输入维度
            hidden_size=config.hidden_size_lstm,  # 隐藏状态维度
            num_layers=config.num_layers,  # LSTM层数
            bidirectional=True,  # 双向LSTM，输出维度翻倍
            batch_first=True  # 批次维度优先
        )

        # Step 3: 初始化全连接层，将LSTM输出映射到分类类别
        # 变量名: self.fc
        # 核心函数: nn.Linear
        # 输入: 池化后的隐藏状态 [batch_size, hidden_size_lstm * 2]
        # 输出: 分类logits [batch_size, class_num]
        self.fc = nn.Linear(
            in_features=config.hidden_size_lstm * 2,  # 双向LSTM输出维度
            out_features=config.class_num  # 分类类别数
        )

        # Step 4: 初始化Dropout层，防止过拟合
        # 变量名: self.dropout
        # 核心函数: nn.Dropout
        # 输入: 隐藏状态 [batch_size, hidden_size_lstm * 2]
        # 输出: 随机丢弃后的隐藏状态 [batch_size, hidden_size_lstm * 2]
        self.dropout = nn.Dropout(p=config.dropout)  # 丢弃概率

    def forward(self, input_ids, attention_mask):
        """
        前向传播，处理输入序列，输出分类logits。

        Args:
            input_ids (torch.Tensor): 输入token ID，维度 [batch_size, seq_len]
            attention_mask (torch.Tensor): 注意力掩码，1表示有效token，0表示padding，维度 [batch_size, seq_len]

        Returns:
            logits (torch.Tensor): 分类logits，维度 [batch_size, class_num]
        """
        # Step 1: 生成有效token掩码，过滤[CLS]和[SEP]
        # 变量名: valid_mask
        # 核心函数: torch.ne, torch.logical_and
        # 输入: input_ids [batch_size, seq_len], attention_mask [batch_size, seq_len]
        # 输出: valid_mask [batch_size, seq_len]，布尔张量，True表示有效token
        cls_token_id = 101  # [CLS] token ID，101
        sep_token_id = 102  # [SEP] token ID，102
        cls_sep_mask = (input_ids != cls_token_id) & (input_ids != sep_token_id)  # 过滤[CLS]和[SEP]
        valid_mask = attention_mask & cls_sep_mask  # 结合注意力掩码，屏蔽padding

        # Step 2: 扩展掩码维度，用于嵌入层屏蔽
        # 变量名: valid_mask_embed
        # 核心函数: torch.unsqueeze
        # 输入: valid_mask [batch_size, seq_len]
        # 输出: valid_mask_embed [batch_size, seq_len, 1]，用于广播操作
        valid_mask_embed = valid_mask.unsqueeze(-1)  # 增加维度以匹配嵌入张量

        # Step 3: 生成嵌入向量，并屏蔽无效token
        # 变量名: embed
        # 核心函数: self.embedding, element-wise multiplication (*)
        # 输入: input_ids [batch_size, seq_len], valid_mask_embed [batch_size, seq_len, 1]
        # 输出: embed [batch_size, seq_len, embed_size]，[CLS]、[SEP]和padding的嵌入置为0
        embed = self.embedding(input_ids)  # 获取原始嵌入
        embed = embed * valid_mask_embed  # 屏蔽无效token的嵌入

        # Step 4: 双向LSTM处理序列，提取特征
        # 变量名: lstm_out
        # 核心函数: self.lstm
        # 输入: embed [batch_size, seq_len, embed_size]
        # 输出: lstm_out [batch_size, seq_len, hidden_size_lstm * 2]，包含所有时间步的隐藏状态
        lstm_out, _ = self.lstm(embed)  # 忽略隐藏状态(h_n, c_n)

        # Step 5: 再次扩展掩码维度，用于LSTM输出屏蔽
        # 变量名: valid_mask_out
        # 核心函数: torch.unsqueeze
        # 输入: valid_mask [batch_size, seq_len]
        # 输出: valid_mask_out [batch_size, seq_len, 1]，用于广播操作
        valid_mask_out = valid_mask.unsqueeze(-1)  # 增加维度以匹配LSTM输出

        # Step 6: 屏蔽LSTM输出的无效token
        # 变量名: masked_output
        # 核心函数: element-wise multiplication (*)
        # 输入: lstm_out [batch_size, seq_len, hidden_size_lstm * 2], valid_mask_out [batch_size, seq_len, 1]
        # 输出: masked_output [batch_size, seq_len, hidden_size_lstm * 2]，[CLS]、[SEP]和padding的隐藏状态置为0
        # 注: 此步骤可能冗余，因嵌入层已屏蔽无效token
        masked_output = lstm_out * valid_mask_out

        # Step 7: 最大池化，提取有效token的最强特征
        # 变量名: hidden
        # 核心函数: torch.max
        # 输入: masked_output [batch_size, seq_len, hidden_size_lstm * 2]
        # 输出: hidden [batch_size, hidden_size_lstm * 2]，序列的最大隐藏状态
        hidden, _ = masked_output.max(dim=1)  # 沿序列维度取最大值

        # Step 8: 应用Dropout，防止过拟合
        # 变量名: hidden
        # 核心函数: self.dropout
        # 输入: hidden [batch_size, hidden_size_lstm * 2]
        # 输出: hidden [batch_size, hidden_size_lstm * 2]，部分神经元随机置0
        hidden = self.dropout(hidden)

        # Step 9: 全连接层，生成分类logits
        # 变量名: logits
        # 核心函数: self.fc
        # 输入: hidden [batch_size, hidden_size_lstm * 2]
        # 输出: logits [batch_size, class_num]，分类logits
        logits = self.fc(hidden)

        return logits

if __name__ == '__main__':
    # 1.加载配置文件
    conf = Config()
    # 2.实例化模型
    model = BiLSTMClassifier(conf)
    # 3.加载数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    print(model)
    # 4.遍历批次，模型预测
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        logits = model(input_ids, attention_mask)
        print(f"Logits shape: {logits}")
