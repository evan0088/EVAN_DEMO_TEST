import numpy as np
import torch

# 矩阵的乘法运算 已知A(n,m)和B(m,p)
# 上述A列=B行, 则A和B可以相乘. 结果是C(n,p)
# 创建张量
A = torch.tensor([[1, 2], [3, 4], [5, 6]])  # 3,2
B = torch.tensor([[5, 6], [7, 8]])  # 2,2
print(A)
print(B)
print('=================================================')
#  TODO 张量(矩阵)乘法运算 @和matmul
# 方式1: @
print(A @ B)
print('-----------------------')
# 方式2: torch.matmul
print(torch.matmul(A, B))
print('-----------------------')
# TODO 张量中有dot()函数,但是只能用于一维张量!!!
# C3 = torch.dot(A, B) # todo 报错,因为torch中dot只支持1维!!!
print(A[0])
print(B[0])
print(torch.dot(A[0], B[0]))
print('=================================================')
# TODO 回顾numpy的矩阵乘法运算
AA = np.array([[1, 2], [3, 4], [5, 6]])
BB = np.array([[5, 6], [7, 8]])
print(AA @ BB)
print('-----------------------')
print(np.matmul(AA, BB))
print('-----------------------')
print(np.dot(AA, BB))  # todo numpy中dot可以用于2维!!!
print('-----------------------')
print(np.dot(AA[0], BB[0]))  # numpy中dot可以用于1维!!!
