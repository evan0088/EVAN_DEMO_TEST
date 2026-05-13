from encoder import *


# 解码器层类 DecoderLayer 实现思路分析
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_p):
        """
        解码器层
        :param size: 等同于d_model
        :param self_attn: 多头自注意力层
        :param src_attn: 编码器-解码器多头一般注意力层
        :param feed_forward: 前馈全连接层
        :param dropout_p: 置零概率
        """
        super(DecoderLayer, self).__init__()
        # 词嵌入维度尺寸大小
        self.size = size
        # 自注意力机制层对象 q=k=v
        self.self_attn = self_attn
        # 一般注意力机制对象 q!=k=v
        self.src_attn = src_attn
        # 前馈全连接层对象
        self.feed_forward = feed_forward
        # clones3子层连接结构
        self.sublayer = clones(SublayerConnection(self.size, dropout_p), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        前向传播
        :param x: 解码器输入层的输出结果 y的word_embedding+positional_encoding
        :param memory: 编码器的输出结果
        :param source_mask: 编码器输入的填充掩码
        :param target_mask: 解码器的自回归掩码&解码器输入的填充掩码
        :return:
        """
        # x->q
        m = memory  # k=v
        # 数据经过子层连接结构1
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 数据经过子层连接结构2
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        # 数据经过子层连接结构3
        x = self.sublayer[2](x, self.feed_forward)
        return x

# 测试
def dm_test_DecoderLayer():
    vocab = 1000  # 词表大小是1000
    d_model = 512

    # 输入x 形状是2 x 4
    # 源数据与目标数据相同, 实际中并不相同
    # x=y
    source = target = torch.LongTensor([[100, 2, 421, 508], [491, 998, 0, 0]])

    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(target)  # [2, 4, 512]

    dropout_p = 0.2  # 置0概率为0.2
    max_len = 60  # 句子最大长度
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embedded_result)

    # 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
    head = 8
    d_ff = 64
    size = 512
    self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout_p)

    # 前馈全连接层也和之前相同
    my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    # 产生编码器结果 k和v
    # 注意此函数返回编码以后的结果 要有返回值, dm_test_Encoder函数后return en_result
    en_result = dm_test_Encoder()

    # 编码器-解码器多头注意力子层填充掩码
    source_mask = (source != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    # 解码器多头自注意力填充掩码
    target_padding_mask = (target != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    # print('target_padding_mask--->', target_padding_mask)
    # 解码器多头自注意力因果掩码
    target_causal_mask = torch.tril(torch.ones(size=(4, 4))).type(torch.uint8).unsqueeze(0).unsqueeze(0)
    # print('target_causal_mask--->', target_causal_mask)
    # 解码器多头自注意力子层掩码
    target_mask = target_padding_mask & target_causal_mask
    # print('target_mask--->', target_mask)

    # 实例化解码器层 对象
    my_dl = DecoderLayer(size, self_attn, src_attn, my_ff, dropout_p)

    # 对象调用
    dl_result = my_dl(pe_result, en_result, source_mask, target_mask)

    print('dl_result.shape--->', dl_result.shape)
    print('dl_result--->', dl_result)



# 解码器类 Decoder 实现思路分析
class Decoder(nn.Module):

    def __init__(self, layer, N):
        # 参数layer 解码器层对象
        # 参数N 解码器层对象的个数
        super(Decoder, self).__init__()
        # clones N个解码器层
        self.layers = clones(layer, N)

    def forward(self, x, memory, source_mask, target_mask):
        # 数据以此经过各个子层
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return x


# 测试 解码器
def dm_test_Decoder():
    vocab = 1000  # 词表大小是1000
    d_model = 512

    # 输入x 形状是2 x 4
    # 源数据与目标数据相同, 实际中并不相同
    source = target = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(target)  # [2, 4, 512]

    dropout_p = 0.2  # 置0概率为0.2
    max_len = 60  # 句子最大长度
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(embedded_result)

    # 分别是解码器层layer和解码器层的个数N
    size = 512
    d_model = 512
    head = 8
    d_ff = 64
    dropout_p = 0.2

    # 多头注意力对象
    self_attn = src_attn = MultiHeadedAttention(head, d_model)

    # 前馈全连接层
    my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    # 产生编码器结果
    en_result = dm_test_Encoder()

    # 编码器-解码器多头注意力子层填充掩码
    source_mask = (source != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    # 解码器多头自注意力填充掩码
    target_padding_mask = (target != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    # 解码器多头自注意力因果掩码
    target_causal_mask = torch.tril(torch.ones(size=(4, 4))).type(torch.uint8).unsqueeze(0).unsqueeze(0)
    # 解码器多头自注意力子层掩码
    target_mask = target_padding_mask & target_causal_mask

    # 创建深拷贝函数
    c = copy.deepcopy
    # 解码器层
    # c(attn):调用深拷贝函数对attn进行深拷贝
    my_dl = DecoderLayer(size, c(self_attn), c(src_attn), c(my_ff), dropout_p)
    N = 6

    # 创建解码器对象
    my_de = Decoder(my_dl, N)

    # 解码器对象 解码
    de_result = my_de(pe_result, en_result, source_mask, target_mask)
    print(de_result)
    print(de_result.shape)
    return de_result


if __name__ == '__main__':
    # 解码器层
    dm_test_DecoderLayer()
    # 解码器
    # de_result = dm_test_Decoder()