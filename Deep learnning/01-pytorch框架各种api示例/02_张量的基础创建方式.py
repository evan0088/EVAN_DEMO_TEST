# 导包
import torch
import numpy as np

# TODO 1.tensor(data,dtype)创建张量: 可以指定数据和类型
print(torch.tensor(10))  # 10   此处10是数据
print(torch.tensor(10).ndim)  # 1   此处10是数据
print(torch.tensor([10]).ndim)  # 1 此处是维度
print(torch.tensor([[10]]).ndim)  # 2 此处是维度
print(torch.tensor([[[10]]]).ndim)  # 3 此处是维度
# 通过列表创建张量
print(torch.tensor([[1, 2, 3], [4, 5, 6]]))
print(torch.tensor([[1, 2, 3], [4, 5, 6]]).dtype)  # 默认torch.int64
# 通过numpy创建张量
print(torch.tensor(np.array([[1, 2, 3], [4, 5, 6]])))  # dtype=torch.int32
print('---------------------------------------------------')
# dtype 可以直接制定类型
print(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32))
print(torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float32))
print('====================================================================')
# TODO 2.Tensor(data/size)创建张量: 注意: 不带[]默认是形状  注意:Tensor没有dtype参数
print(torch.Tensor(10))  # 此处10是形状 默认数据是0.
print(torch.Tensor(10).ndim)  # 此处10是形状 默认数据是0.
print(torch.Tensor([10]).ndim)
print(torch.Tensor([[10]]).ndim)
print(torch.Tensor([[[10]]]).ndim)
print(torch.Tensor(2, 3))  # 此处2,3是形状 默认数据是0.
print(torch.Tensor(size=(2, 3)))
# 通过列表创建张量
print(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
print(torch.Tensor([[1, 2, 3], [4, 5, 6]]).dtype)  # 默认torch.float32
# 通过numpy创建张量
print(torch.Tensor(np.array([[1, 2, 3], [4, 5, 6]])))
print(torch.Tensor(np.array([[1, 2, 3], [4, 5, 6]])).dtype)  # torch.float32
print('---------------------------------------------------')
# 主要区别总结：
# torch.Tensor() - 不自动推断类型，默认创建float32类型，没有dtype参数
# torch.tensor() - 自动推断类型，可以根据输入数据自动确定类型，支持dtype参数指定类型
# 推荐使用torch.tensor()，因为它更灵活且能自动推断类型
# TODO 3.IntTensor()/FloatTensor()/LongTensor()/ShortTensor()/ByteTensor()/HalfTensor()创建张量
print(torch.ByteTensor([[1, 2, 3], [4, 5, 6]]))
print(torch.ShortTensor([[1, 2, 3], [4, 5, 6]]))
print(torch.IntTensor([[1, 2, 3], [4, 5, 6]]))
print(torch.LongTensor([[1, 2, 3], [4, 5, 6]]))  # 默认int64
print(torch.HalfTensor([[1, 2, 3], [4, 5, 6]]))
print(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]))  # 默认float32
print(torch.DoubleTensor([[1, 2, 3], [4, 5, 6]]))
