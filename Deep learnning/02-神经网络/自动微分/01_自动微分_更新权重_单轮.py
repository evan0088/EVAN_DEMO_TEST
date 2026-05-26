# 导包
import torch

# TODO 自动微分模块主要用于梯度计算!!!
# 定义权重 requires_grad=True:开启自动微分
w = torch.tensor([10, 20], requires_grad=True, dtype=torch.float)
# 自定义损失函数
loss = 2 * w ** 2
# todo 方式1: 手动算梯度  2w² 导数是4x
# 带入x=10 ,4*10=40
# 带入x=20 ,4*20=80
# todo 方式2: 自动微分求梯度,自动更新梯度
# 调用backward()必须是一个标量,如果不是就需要sum()或者mean()把多个值聚合成一个标量
loss.sum().backward()
# 格式化输出
print(f"当前权重: {w.data},固定学习率:{0.01} 更新后梯度: {w.grad},下一个权重: {w.data - 0.01 * w.grad}")
# TODO 后续大模型训练的时候,就可以拿着刚刚的梯度去手动更新权重了,继续下一次
# 公式:  w1 = w0 - learning_rate * grad
w.data = w.data - 0.01 * w.grad
