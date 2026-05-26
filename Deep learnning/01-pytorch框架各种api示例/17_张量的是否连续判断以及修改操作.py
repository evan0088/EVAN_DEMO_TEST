import torch

# 创建张量
t1 = torch.tensor([[10, 20, 30], [40, 50, 60]])
print(t1, t1.shape, t1.ndim)
print('-----------------------------')
# print(t1.reshape(3, 2))  # reshape()修改后是连续的
t2 = t1.transpose(1, 0)
print(t2, t2.shape, t2.ndim) # transpose()修改是不连续的
print('========================================')
# TODO is_contiguous()判断张量是否连续
# 判断t1是否连续
print(t1.is_contiguous()) # True 连续
# 判断t2是否连续
print(t2.is_contiguous()) # False 不连续
print('========================================')
# TODO 演示reshape和view的区别:
#  todo 1.reshape连续和不连续张量都能操作,以后工作中常用reshape()
print(t1.reshape(1, 6))
print(t2.reshape(1, 6))
print('--------------------------------')
#  todo 2.view只能操作连续张量,如果是不连续的怎么办? 使用contiguous()
print(t1.view(1, 6))
# print(t2.view(1, 6))  # 报错,因为t2是不连续的
# TODO 使用contiguous()把数据变为连续的,然后使用view()
t3 = t2.contiguous()
print(t3)
print(t3.is_contiguous()) # True 连续
print(t3.view(1, 6))  # 成功,因为t3是修改后连续的
