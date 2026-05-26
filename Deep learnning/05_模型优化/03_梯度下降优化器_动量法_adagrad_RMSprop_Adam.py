# 导包
import torch

# 定义权重 requires_grad=True:开启自动微分
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
# TODO 优化器
# TODO 方式1: SGD+动量法: 核心思想就是指数加权平均优化梯度
# optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)
# TODO 方式2: Adagrad: 核心思想是学习率除以梯度平方和+小常数(弊端是学习率过早衰减)
# optimizer = torch.optim.Adagrad([w], lr=0.01)
# TODO 方式3: RMSprop: 核心思想就是基于Adagrad引入了指数加权平均(缓解学习率过早衰减)
# optimizer = torch.optim.RMSprop([w], lr=0.01,alpha=0.99)
# TODO 方式4: Adam: 核心思想就是动量法+RMSprop
optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.99))
print(f"初始权重: {w}")
# 循环迭代权重
for i in range(50):
    # 定义损失函数
    loss = w ** 2 / 2.0
    # 梯度清零!!!
    optimizer.zero_grad()
    # 反向传播:梯度计算
    loss.backward()
    # 参数更新
    optimizer.step()  # 底层就是w1 = w0-lr*梯度
    # 打印日志
    print(f"更新后权重: {w}")
