import torch

# 创建张量
t = torch.tensor([1, 2, 3, 4, 5, 6])
print(t, t.shape, t.ndim) # 形状(6)
print('==============================================')
# TODO 先使用reshape()变向完成升维操作,可以操作连续和非连续的张量
t2 = t.reshape(2, 3)
print(t2, t2.shape, t2.ndim)
t3 = t.reshape(1, 2, 3)
print(t3, t3.shape, t3.ndim)
print('----------------------------')
# TODO 再使用专业的升维操作: unsqueeze()   (0,1)和(-2,-1)正负两套索引
t4 = t.unsqueeze(dim=0)  # 形状(1,6)
print(t4, t4.shape, t4.ndim)
t5 = t.unsqueeze(dim=1)  # 形状(6,1)
print(t5, t5.shape, t5.ndim)
# tt = t.unsqueeze(dim=2) # 报错: 维度超出范围（预期应在[-2, 1]范围内，但得到的是 2）
tt = t.unsqueeze(dim=-1) # 形状(6,1)
print(tt, tt.shape, tt.ndim)
tt = t.unsqueeze(dim=-2) # 形状(1,6)
print(tt, tt.shape, tt.ndim)
print('==============================================')
# TODO 使用专业的降维操作: squeeze()
t6 = t4.squeeze()
print(t6, t6.shape, t6.ndim)
t7 = t5.squeeze()
print(t7, t7.shape, t7.ndim)
