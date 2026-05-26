# 导包
import torch
from matplotlib import pyplot as plot

"""
核心知识: 
当前指数加权平均 = 历史权重 * 历史指数加权平均 + (1-历史权重)*当前数据
结论: 历史权重越大越平缓!!!
"""

# TODO 需求: 绘制30天的温度数据图(不使用指数加权平均)
# x轴天:1-30
x = torch.arange(1, 31)
# 提前设置种子
torch.manual_seed(0)
# y轴温度:随机
y = torch.randn([30, ]) * 10
# 画图
plot.scatter(x, y)
plot.plot(x, y, color='red')
plot.show()
print('====================================================')
# TODO 需求: 绘制30天的温度数据图(使用指数加权平均)
# x轴天:1-30
x = torch.arange(1, 31)
# 提前设置种子
torch.manual_seed(0)
# y轴温度:随机
y = torch.randn([30, ]) * 10
# todo 生成指数加权平均列表
y_ewa = []
beta = 0.9
for idx, t in enumerate(y, 1):
    if idx == 1:
        y_ewa.append(t)
        continue
    # 从第2个开始就需要指数加权平均计算,然后放到列表中
    ewa = beta * y_ewa[-1] + (1 - beta) * t
    y_ewa.append(ewa)
# 画图
plot.scatter(x, y_ewa)
plot.plot(x, y_ewa, color='red')
plot.show()
