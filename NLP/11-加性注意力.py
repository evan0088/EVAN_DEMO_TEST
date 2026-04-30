# 加性注意力
import torch
import torch.nn as nn


# 创建注意力机制模型类
class MyAttn(nn.Module):
    # todo:1-构造方法 初始化属性
    def __init__(self, query_size, key_size, seq_len, hidden_size, output_size):
        super().__init__()
        # 解码器上一个时间步输出或当前时间步输入的维度
        self.query_size = query_size
        # 编码器hn或解码器上一个时间步隐藏状态值的维度
        self.key_size = key_size
        # 句子长度, 编码器总时间步, token数
        self.seq_len = seq_len
        # 编码器隐藏层的维度
        self.hidden_size = hidden_size
        # q和动态c融合后的维度数 = 后续rnn/lstm/gru/transformer的输入维度
        self.output_size = output_size

        # 线性层1: q和k在特征维度拼接后进行线性计算
        # in_features: q特征数+k特征数
        # out_features: v的词个数/句子长度 seq_len  为什么?  softmax结果和v进行三维矩阵乘法  (1,1,句子长度)*(句子数,句子长度,词维度)
        # 将(1,1,64)经过线性层1得到(1,1,32)
        self.attn = nn.Linear(in_features=self.query_size+self.key_size,
                              out_features=self.seq_len)

        # 线性层2: q和c在特征维度拼接然后进行线性计算
        # c:attn_weights*v=(1,1,seq_len)*(1,seq_len,hidden_size)=(1,1,hidden_size)
        # in_features: q特征数+v特征数
        # out_features: 后续gru/lstm/transformer的要求输入维度
        self.attn_combine = nn.Linear(in_features=self.query_size+self.hidden_size,
                                      out_features=self.output_size)
        print("query_size+hidden_size --->",self.query_size+self.hidden_size)

        # 后续步骤, 初始化gru层
        self.gru = nn.GRU(input_size=self.output_size, hidden_size=512, num_layers=1, batch_first=True)

    # todo:2-前向传播方法 forward
    def forward(self, query, key, value):
        # todo:2-1 计算q和k的相似分, q和k在特征维度拼接后经过线性层1映射成相似分
        q_k_cat = torch.cat(tensors=[query, key], dim=-1)
        print('q_k_cat.shape--->', q_k_cat.shape, q_k_cat)
        # 加性注意力是通过线性层计算相似分
        attn_scores = self.attn(q_k_cat)
        print('attn_scores--->', attn_scores.shape, attn_scores)

        # todo:2-2 softmax计算得到权重矩阵
        attn_weights = torch.softmax(input=attn_scores, dim=-1)
        print('attn_weights--->', attn_weights.shape, attn_weights, attn_weights.sum(dim=-1))

        # todo:2-3 计算加性注意力, 得到动态c
        # (1,1,32)*(1,32,32)=(1,1,32)
        print('value--->', value.shape, value)
        attn_c = torch.bmm(input=attn_weights, mat2=value)
        print('attn_c--->', attn_c.shape, attn_c)

        # ============= 以上就是加性注意力计算的结果 =============
        # 后续将q和动态c融合给到下一步(gru/lstm/transformer隐层)
        attn_q_c = torch.cat(tensors=[query, attn_c], dim=-1)
        print('attn_q_c--->', attn_q_c.shape, attn_q_c)
        # 经过线性层2, 得到最终融合的结果=gru层的输入x
        input_x = self.attn_combine(attn_q_c)
        print('input_x--->', input_x.shape, input_x)
        # 调用gru层, input_x作为输入x
        output, hn = self.gru(input_x)
        print('output--->', output.shape, output)





if __name__ == '__main__':
    # q的特征维度
    query_size = 32
    # k的特征维度
    key_size = 32
    # 句子长度, 词数量
    seq_len = 32
    # 隐层特征维度数
    hidden_size = 32
    # q和动态c融合后的特征维度数
    output_size = 32

    # 定义q
    # (1个句子, 1个时间步, 维度数)
    q = torch.randn(1, 1, query_size)
    # 定义k 解码器第1个时间步的初始隐藏状态值=编码器的hn k和q形状相同, 要进行concat拼接
    # 如何不同 (1, 3, key_size), 需要在1轴上进行处理
    k = torch.randn(1, 1, key_size)
    # 定义v, 编码器的output  hidden_size=key_size
    v = torch.randn(1, seq_len, hidden_size)

    # 实例化加性注意力模型类对象
    my_attn = MyAttn(query_size=query_size,
                     key_size=key_size,
                     seq_len=seq_len,
                     hidden_size=hidden_size,
                     output_size=output_size)
    my_attn(query=q, key=k, value=v)
