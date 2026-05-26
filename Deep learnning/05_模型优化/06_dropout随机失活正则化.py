# 导包
import torch

# 创建隐藏层
linear1 = torch.nn.Linear(4, 5)
# 准备输入数据
x = torch.randint(1, 10, (1, 4), dtype=torch.float)
print("输入:",x)
# 加权求和->激活函数
x = linear1(x)
print("加权求和:",x)
x = torch.relu(x)
print("relu激活:",x)
# TODO 创建dropout层: 在激活层之后, 创建dropout层
dropout = torch.nn.Dropout(p=0.5)
# 随机失活
x = dropout(x)
print("dropout随机失活:",x)
