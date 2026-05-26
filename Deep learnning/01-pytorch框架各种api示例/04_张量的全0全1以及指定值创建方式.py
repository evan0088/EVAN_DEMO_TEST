import torch

# TODO 1.创建全0张量
# zeros(size) 创建全0张量
print(torch.zeros(2, 3))
print(torch.zeros(4, 5))
# zeros_like(张量) 模仿指定张量的形状创建全0张量
# 提前创建一个形状为(2, 3)的张量
x = torch.randint(0, 10, size=(2, 3))
print(x)
print(torch.zeros_like(x))
print('---------------------------------------')
# TODO 2.创建全1张量
# ones(size) 创建全1张量
print(torch.ones(2, 3))
print(torch.ones(4, 5))
# ones_like(张量) 模仿指定张量的形状创建全1张量
# 提前创建一个形状为(2, 3)的张量
x = torch.randint(0, 10, (2, 3))
print(x)
print(torch.ones_like(x))
print('---------------------------------------')
# TODO 3.创建全指定值张量
# full(size,value) 创建全1张量
print(torch.full((2, 3), 8))
print(torch.full((4, 5), 9))

# full_like(张量) 模仿指定张量的形状创建全指定值张量
# 提前创建一个形状为(2, 3)的张量
x = torch.randint(0, 10, (2, 3))
print(x)
print(torch.full_like(x, 9))
