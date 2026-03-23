# 导包
import torch

# TODO 创建张量的时候,开启自动微分
x = torch.tensor([2.0], requires_grad=True)
# 定义函数
y = x ** 3
# 求导数(梯度)
# 方式1: 手动求导(梯度)
# todo x ** 3的求导是3x²,依次带入x=2.0,导数是12
# 方式2: 自动微分自动求导(梯度)
y.backward()
print(x.grad)  # tensor([12.])
print('===================================================')
# TODO 创建张量的时候,开启自动微分
x = torch.tensor([2.0, 3.0], requires_grad=True)
# 定义函数
y = x ** 3
# 求导数(梯度)
# 方式1: 手动求导(梯度)
# todo x ** 3的求导是3x²,依次带入x=2.0和3.0,导数是12和27
# 方式2: 自动微分自动求导(梯度)
y.sum().backward()  # TODO 此处y必须是标量才能调用backward()
print(x.grad)  # tensor([12., 27.])
