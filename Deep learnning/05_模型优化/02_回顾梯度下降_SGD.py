# 导包
import torch

# 定义权重 requires_grad=True:开启自动微分
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
# TODO SGD优化器
optimizer = torch.optim.SGD([w], lr=0.01)
print(f"初始权重: {w}")
# 循环迭代权重
for i in range(5):
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

