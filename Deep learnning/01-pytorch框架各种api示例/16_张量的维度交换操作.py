import torch

# 提前设置一个种子
torch.manual_seed(666)
# 创建三维张量
t = torch.randint(1, 5, size=(3, 4, 5))
print(t, t.shape)
#                    0 1 2       1 2 0
# TODO 需求: 把张量形状(3,4,5)转变为(4,5,3)
print('=====================================================')
# TODO transpose方式: 交换多次,因为它一次只能交换两个
t2 = t.transpose(1, 0)  # (3,4,5)->(4,3,5)
print(t2, t2.shape)
print('------------------------------------------------')
t3 = t2.transpose(2, 1)  # (4,3,5)->(4,5,3)
print(t3, t3.shape)
print('=====================================================')
# TODO permute方式: 一次指定多个维度
t4 = t.permute(dims=(1, 2, 0))  # (3,4,5)->(4,5,3)
print(t4, t4.shape)
# 注意: 还可以直接使用torch调用transpose()和permute()
