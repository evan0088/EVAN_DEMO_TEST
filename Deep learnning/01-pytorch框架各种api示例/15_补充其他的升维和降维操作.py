import torch

# 创建张量
# 创建张量
t = torch.tensor([1, 2, 3, 4, 5, 6])
print(t, t.shape, t.ndim)
print('==============================================')
# TODO 其他升维操作
#  todo 1 使用类似reshape()的view()函数,不同点是view()只能操作连续的张量
t2 = t.view(2, 3)
print(t2, t2.shape, t2.ndim)
t3 = t.view(1, 2, 3)
print(t3, t3.shape, t3.ndim)
print('==============================================')
#  todo 2 利用索引
t4 = t[None, :]
print(t4, t4.shape, t4.ndim)
t5 = t[:, None]
print(t5, t5.shape, t5.ndim)
t6 = t[None, :, None]
print(t6, t6.shape, t6.ndim)
print('==============================================')
print(t6, t6.shape, t6.ndim)
# TODO 注意: squeeze()默认删除所有1维,完成降维操作
t7 = t6.squeeze()
print(t7, t7.shape, t7.ndim)
# TODO 创建一个高维的张量,然后完成降维操作
t8 = torch.randint(1, 5, size=(1, 2, 1, 1, 5, 1, 3))
# t9 = torch.squeeze(t8)
# print(t9.shape, t9.ndim)
print(t8.shape,t8.ndim)
t8.squeeze_()
print(t8.shape,t8.ndim)
