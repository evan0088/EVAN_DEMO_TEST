"""
目的:
    通过代码 观察不同的学习率 对 参数更新 的影响.
"""

import torch
import matplotlib.pyplot as plt


# 损失函数
def func(x_t):
    return torch.pow(2 * x_t, 2)  # y = 4 x ^2


# x看成是权重，y看成是loss，下面通过代码来理解学习率的作用
x = torch.tensor([2.], requires_grad=True)
# 记录loss迭代次数，画曲线
iter_rec, loss_rec, x_rec = list(), list(), list()
# TODO 调整学习率，观察结果
lr = 0.05
"""
0.01及以下: 梯度下降慢
0.05-0.1: 梯度下降正常
0.125: 类似正归方程一步到位
0.2: 梯度震荡
0.3及以上: 梯度爆炸
调整上述lr学习率,结论如下:
    较小的学习率：梯度下降速度慢，收敛时间长，但相对稳定。
    适当的学习率：梯度下降速度快，收敛时间短，且能够稳定收敛到最小值点。
    较大的学习率：梯度下降速度过快，容易越过最小值点，导致震荡甚至,梯度爆炸。
"""
max_iteration = 4
for i in range(max_iteration):
    y = func(x)  # 得出loss值
    y.backward()  # 计算x的梯度
    print("Iter:{}, X:{:8}, X.grad:{:8}, loss:{:10}".format(
        i, x.detach().numpy()[0], x.grad.detach().numpy()[0], y.item()))
    x_rec.append(x.item())  # 梯度下降点 列表
    # 更新参数
    x.data.sub_(lr * x.grad)  # x = x - x.grad
    x.grad.zero_()
    iter_rec.append(i)  # 迭代次数 列表
    loss_rec.append(y.item())  # 损失值 列表，这里将y改为y.item()以获取标量值
# 迭代次数-损失值 关系图
plt.subplot(121).plot(iter_rec, loss_rec, '-ro')
plt.grid()
plt.xlabel("Iteration X")
plt.ylabel("Loss value Y")
# 函数曲线-下降轨迹 显示图
x_t = torch.linspace(-3, 3, 100)
y = func(x_t)
plt.subplot(122).plot(x_t.detach().numpy(), y.detach().numpy(), label="y = 4*x^2")
y_rec = [func(torch.tensor(i)).item() for i in x_rec]
print('x_rec--->', x_rec)
print('y_rec--->', y_rec)
# 指定线的颜色和样式（-ro：红色圆圈，b-：蓝色实线等）
plt.subplot(122).plot(x_rec, y_rec, '-ro')
plt.grid()
plt.legend()
plt.show()
