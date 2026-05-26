# 导包
import torch

# 主要解决内部协变量偏移问题? 将输入特征减去均值除以标准差,使得数据均值0,方差1,避免内部协变量偏移
# TODO 创建BN层对象
# affine=True : 是否可学习参数权重和偏置
bn2d = torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
# 准备数据  解释: (N,C,H,W) 1个2通道的3*4像素的图片
input = torch.randn(1, 2, 3, 4)
print(input)
# TODO 归一化处理
output = bn2d(input)
print(output)
# 打印参数
print(bn2d.weight)
print(bn2d.bias)
