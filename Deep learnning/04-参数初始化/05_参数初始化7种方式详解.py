# 导包
import torch

# TODO 1.随机均匀初始化
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.uniform_(linear1.weight)
print(linear1.weight)
print(linear1.bias)
print('=========================================')
# TODO 2.正态分布初始化  注意: 传入均值和标准差
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.normal_(linear1.weight, mean=0, std=1)
print(linear1.weight)
print(linear1.bias)
print('=========================================')
# TODO 3.全0初始化
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.zeros_(linear1.weight)
print(linear1.weight)
print(linear1.bias)
print('=========================================')
# TODO 4.全1初始化
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.ones_(linear1.weight)
print(linear1.weight)
print(linear1.bias)
print('=========================================')
# TODO 5.固定值初始化 注意:传入固定值
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.constant_(linear1.weight, 2.6)
print(linear1.weight)
print(linear1.bias)
print('=========================================')
# TODO 6.kaiming初始化  注意: 正态分布不需均值,标准差
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.kaiming_normal_(linear1.weight)
print(linear1.weight)
print(linear1.bias)
print('----------------------------')
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.kaiming_uniform_(linear1.weight)
print(linear1.weight)
print(linear1.bias)
print('=========================================')
# TODO 7.xavier初始化 注意: 正态分布不需均值,标准差
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.xavier_normal_(linear1.weight)
print(linear1.weight)
print(linear1.bias)
print('----------------------------')
# 创建一个隐藏层
linear1 = torch.nn.Linear(5, 3)
# 随机初始化
torch.nn.init.xavier_uniform_(linear1.weight)
print(linear1.weight)
print(linear1.bias)
