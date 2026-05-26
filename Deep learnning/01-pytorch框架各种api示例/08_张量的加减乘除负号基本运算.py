import torch

# TODO 张量的加减乘除负号基本运算
t1 = torch.tensor([[1, 2], [3, 4]])
print(t1)
print('----------------------')
print(t1 + 2)
print(t1 - 2)
print(t1 * 2)
print(t1 / 2)
print(t1 * -1)
print('----------------------')
print(torch.add(t1, 2))
print(torch.sub(t1, 2))
print(torch.mul(t1, 2))
print(torch.div(t1, 2))
print(torch.neg(t1))
print('----------------------')
# TODO add_,  sub_, mul_, div_, neg_直接修改原有张量数据,但是类型不能变更
print(t1)
t1.add_(2)
print(t1)
t1.sub_(2)
print(t1)
t1.mul_(2)
print(t1)
# t1.div_(2) # 注意:除以2后结果是浮点类型,原来数据是整数类型,此行报错! 因为不能直接变更原有张量元素类型
t1.neg_()
print(t1)


