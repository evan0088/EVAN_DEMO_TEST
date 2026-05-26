"""02_激活函数"""
"""
sigmoid特点
正负样本都考虑,把对应值转变到对应值范围: (0,1)
导数范围:(0,0.25]
用途:可以用于隐藏层,但主要用于输出层,解决二分类问题
缺点:梯度消失问题严重,非零中心化
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
print(x[:5]) # 内部状态值 区间任意
# 计算y轴的值
y = torch.sigmoid(x)
print(y[:5]) # 激活值 区间是0-1
#  绘制图像
ax[0].plot(x, y)
ax[0].set_title('函数图像')
ax[0].grid()
print('---------------------------------------------')
# TODO 3.绘制导数图像
# 准备x轴的值
x = torch.linspace(-20, 20, 1000, requires_grad=True)
# 计算y轴的值
torch.sigmoid(x).sum().backward()
#  绘制图像
ax[1].plot(x.detach(), x.grad)
ax[1].set_title('导数图像')
ax[1].grid()
plt.show()
