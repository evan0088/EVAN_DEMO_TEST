import torch
# 创建张量
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t)
print('----------------------------')
# TODO shape获取形状
print(t.shape, t.shape[0], t.shape[1], t.shape[-1])
print(t.size(), t.size()[0], t.size()[1], t.size()[-1], t.size(-1))
print('===================================')
# TODO reshape修改形状(元素个数不能变化!!!)
print(t.reshape(3, 2))
print('--------------------')
print(t.reshape(1, 6))
print('--------------------')
print(t.reshape(6, 1))
print('--------------------')
# print(t.reshape(2, 2))  # 个数不匹配就报错!!!
# print(t.reshape(3, 3))  # 个数不匹配就报错!!!
