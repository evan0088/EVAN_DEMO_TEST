# 导包
import torch
# 准备x训练数据
x = torch.ones(2, 5)
print('x:', x)
# 准备y训练数据
y = torch.zeros(2, 3)
print('y:', y)
# todo 准备w权重矩阵,开启自动微分!!!
w = torch.randn(5, 3, requires_grad=True, dtype=torch.float)
print('w:', w)
# todo 准备b偏置矩阵,开启自动微分!!!
b = torch.randn(3, requires_grad=True)
print('b:', b)
print('=====================================================')
# TODO 最终目的根据上述数据,使用自动微分推导w和b的梯度
# 1.首先获取损失函数
loss_fn = torch.nn.MSELoss()
# 2.然后,计算预测值-> z=wx+b 注意: 这里面的wx是矩阵乘法需要遵循 (n,m)*(m,p)=(n,p)
# z = x.matmul(w) + b
z = x @ w + b
print(f"z->:{z}")
# 3.接着,根据损失函数计算损失值
loss = loss_fn(z, y)
# 4.最后,反向传播推导更新梯度
loss.sum().backward()
# TODO 打印更新后w和b梯度
print(f'w.grad: {w.grad}')
print(f'b.grad: {b.grad}')










