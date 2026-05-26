import numpy as np
import torch

#  TODO 1.numpy数组转换为张量
# todo from_numpy()将numpy数组转换为张量,但是共享内存
# 创建numpy数组
n1 = np.array([1, 2, 3])
t1 = torch.from_numpy(n1)
print(n1, type(n1))
print(t1, type(t1))
# 演示from_numpy()结果共享内存
n1[0] = 100
print(n1, id(n1))
print(t1, id(t1))
print('------------------------------')
# todo torch.tensor(ndarray)将numpy数组转换为张量,不会共享内存
# 创建numpy数组
n2 = np.array([1, 2, 3])
t2 = torch.tensor(n2)
print(n2, type(n2))
print(t2, type(t2))
# 演示tensor(ndarray)结果不共享内存
n2[0] = 200
print(n2, id(n2))
print(t2, id(t2))
print('========================================================')
#  TODO 2.张量转换为numpy数组
# todo numpy()将张量转换为numpy数组,共享内存
# 创建张量
t3 = torch.tensor([11, 22, 33])
# 转换
n3 = t3.numpy()
print(t3, type(t3))
print(n3, type(n3))
# 演示numpy()结果共享内存
t3[0] = 300
print(t3, id(t3))
print(n3, id(n3))
print('------------------------------')
# todo numpy().copy()将张量转换为numpy数组,不共享内存
# 创建张量
t4 = torch.tensor([11, 22, 33])
# 转换
n4 = t4.numpy().copy()
print(t4, type(t4))
print(n4, type(n4))
# 演示numpy().copy()结果不共享内存
t4[0] = 300
print(t4, id(t4))
print(n4, id(n4))
