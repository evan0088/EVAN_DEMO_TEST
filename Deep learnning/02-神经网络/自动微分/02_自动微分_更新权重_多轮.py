# 导包
import torch

# TODO 自动微分模块主要用于梯度计算!!!
# 定义权重 requires_grad=True:开启自动微分
w = torch.tensor(10, requires_grad=True, dtype=torch.float)
# 打印首次默认梯度
print(f"初始权重: {w.data},初始梯度: {w.grad}")  # 初始梯度None
# TODO 定义遍历轮次
epochs = 100
# TODO 开始遍历
for epoch in range(epochs):
    # 自定义损失函数
    # loss = w ** 2 + 20
    loss = 2 * w ** 2
    # TODO 注意: 默认梯度是累加的,所以每个轮次需要在自动微分之前进行清零!!!
    if w.grad is not None:
        w.grad.zero_()  # 后续使用优化器清零
    # 自动微分求梯度,自动更新梯度
    loss.sum().backward()
    # 格式化输出
    print(f"当前轮次:{epoch + 1} 当前权重: {w.data},固定学习率:{0.01} 更新后梯度: {w.grad},下一个权重: {w.data - 0.01 * w.grad}")
    # TODO 手动更新权重
    # 公式:  w1 = w0 - learning_rate * grad
    w.data = w.data - 0.01 * w.grad
