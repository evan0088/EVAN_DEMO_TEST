import torch
# 提前设置种子
torch.manual_seed(1)
# 创建三维张量
data = torch.randint(1, 10, size=(3, 4, 5))
print(data)
# print(data[:, :, :])
# 格式: data[0轴索引,1轴索引,2轴索引]
print('----------------------------------')
# 获取0轴上的第一个数据
print(data[0, :, :])
print(data[0])
# 获取0轴上的前2个数据
print(data[0:2])
print('----------------------------------')
# 获取1轴上的第一个数据
print(data[:, 0, :])
# 获取1轴上的前2个数据
print(data[:, 0:2, :])
print('----------------------------------')
# 获取2轴上的第一个数据
print(data[:, :, 0])
# 获取2轴上的前2个数据
print(data[:, :, 0:2])