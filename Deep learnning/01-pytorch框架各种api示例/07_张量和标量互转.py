# 导包
import torch

# TODO 标量转换张量
a1 = 10
t1 = torch.tensor(a1)
print(a1, type(a1))
print(t1, type(t1))
print('------------------------------')
# TODO 张量转换标量
a2 = t1.item()
print(t1, type(t1))
print(a2, type(a2))
print('------------------------------')
# TODO 注意: item()只能应用张量中只有一个元素的情况
t2 = torch.tensor([1, 2, 3])
print(t2.item()) # TODO 报错
