"""02_激活函数"""
"""
relu特点
不考虑负样本,直接把正样本数据转变到对应值范围: [0,正无穷)
导数范围: {0或者1}
用途:主要就是用于隐藏层,且是隐藏层中最常用的!!! 
因为不考虑负样本导致部分神经元死亡,有效缓解过拟合问题
相比sigmoid:有效的缓解了梯度消失问题
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
y = torch.relu(x)
#  绘制图像
ax[0].plot(x, y)
ax[0].set_title('函数图像')
ax[0].grid()

# TODO 3.绘制导数图像
# 准备x轴的值
x = torch.linspace(-20, 20, 1000, requires_grad=True)
# 计算y轴的值
torch.relu(x).sum().backward()
#  绘制图像
ax[1].plot(x.detach(), x.grad)
ax[1].set_title('导数图像')
ax[1].grid()
plt.show()
