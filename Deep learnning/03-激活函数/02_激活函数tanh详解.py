"""02_激活函数"""
"""
tanh特点
正负样本都考虑,把对应值转变到对应值范围: (-1,1)
导数范围:(0,1]
用途:可以用于输出层,但一般用于浅隐藏层
相比sigmoid: 引入了零中心化,但是梯度消失问题依然存在(比sigmoid有所缓解)
"""
# 导包
from matplotlib import pyplot as plt
import torch

# 解决显示中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# TODO 1.创建一行两列的画布
fig, ax = plt.subplots(1, 2)
# TODO 2.绘制函数图像
# 准备x轴的值
x = torch.linspace(-20, 20, 1000)
# 计算y轴的值
y = torch.tanh(x)
#  绘制图像
ax[0].plot(x, y)
ax[0].set_title('函数图像')
ax[0].grid()
print('-------------------------------------------------')
# TODO 3.绘制导数图像
# 准备x轴的值
x = torch.linspace(-20, 20, 1000, requires_grad=True)
# 计算y轴的值
torch.tanh(x).sum().backward()
#  绘制图像
ax[1].plot(x.detach(), x.grad)
ax[1].set_title('导数图像')
ax[1].grid()
plt.show()
