import torch

# 提前设置一个随机种子
torch.manual_seed(666)
# TODO cat()拼接三维案例:对应维度上的维数相加
# 创建张量
t1 = torch.randint(1, 5, (1, 2, 3))
t2 = torch.randint(1, 5, (1, 2, 3))
print(t1)
print(t2)
print('----------------------------')
# 拼接
print(torch.cat([t1, t2], dim=0))  # (2, 2, 3)
print('----------------------------')
# 拼接
print(torch.cat([t1, t2], dim=1))  # (1, 4, 3)
print('----------------------------')
# 拼接
print(torch.cat([t1, t2], dim=2))  # (1, 2, 6)
print('=========================================================')
#  TODO cat()拼接的是除拼接维度外,其他所有张量的形状必须完全相同。
t1 = torch.randint(1, 5, (2, 3))
t2 = torch.randint(1, 5, (1, 3))
print(t1)
print(t2)
# 拼接0轴上
print(torch.cat([t1, t2], dim=0))  # (3, 3)
# 拼接1轴上,但是0轴上的维数不同
# print(torch.cat([t1, t2], dim=1))  # 报错

print('=========================================================')
# TODO stack()拼接的是所有张量的形状必须完全相同（所有维度一致）。
t1 = torch.randint(1, 5, (2, 3))
t2 = torch.randint(1, 5, (2, 3))
print(t1)
print(t2)
print(torch.stack([t1, t2], dim=0))  # (2,2,3)
print(torch.stack([t1, t2], dim=1))  # (2,2,3)
print(torch.stack([t1, t2], dim=2))  # (2,3,2)
