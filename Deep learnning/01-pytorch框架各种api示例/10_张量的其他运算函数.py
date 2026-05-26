# 张量的其他运算函数
import torch

# 设置随机种子(方便使用统一数据)
torch.manual_seed(666)
# 创建张量
t = torch.randint(1, 10, size=(2, 3), dtype=torch.float32)
print(t)
print('----------------------------------------------')
# TODO 演示sum()/mean()/max()/min()/pow()/sqrt()/log()/log2()/log10()/exp()
print(t.sum())
print(t.sum(dim=0))  # dim=0，按列求和
print(t.sum(dim=1))  # dim=1 按行求和
print('----------------------------------------------')
print(t.mean())
print(t.mean(dim=0))  # dim=0 按列求均值
print(t.mean(dim=1))  # dim=1 按行求均值
print('----------------------------------------------')
print(t.max())  # 打印t的最大值
print('----------------------------------------------')
print(t.min())  # 打印t的最小值
print('----------------------------------------------')
print(t.pow(2))  # 打印t的平方
print(t.pow(3))  # 打印t的立方
print('----------------------------------------------')
print(t.sqrt())  # 打印t的平方根
print('----------------------------------------------')

print(t.log2())  # 打印t的以2为底的对数
print(t.log10())  # 打印t的以10为底的对数
print('----------------------------------------------')
print(t.exp())  # 打印t的指数函数结果:以e为底,tensor中每个元素的指数函数结果
print('----------------------------------------------')
# math数学模块
import math

print(math.e)
print(math.pi)
print(math.sqrt(16))
print(math.log(math.e))
print(math.log(math.e, 2))
print(math.log(math.e, 10))
print(math.log10(1000))
print(math.log2(16))
print(math.exp(1))
print(math.pow(2, 3))
