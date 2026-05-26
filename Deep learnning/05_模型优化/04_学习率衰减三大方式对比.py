# 导包
import torch
import matplotlib.pyplot as plt

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# todo 准备x,y数据
x = torch.tensor([1.0], dtype=torch.float)
y_true = torch.tensor([0.0], dtype=torch.float)
# 定义权重 requires_grad=True:开启自动微分
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
# SGD优化器
optimizer = torch.optim.SGD([w], lr=0.1)
# TODO 学习率衰减三种方式
# todo 方式1: 等间隔衰减
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
# todo 方式2: 指定间隔衰减
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 125, 175], gamma=0.5)
# todo 方式3: 指数衰减
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# TODO 为了后面绘制每轮学习率图,此处需要记录轮次以及对应学习率
epoch_list = []
lr_list = []
# todo 外层控制训练轮次200轮
for epoch in range(1, 201):
    # TODO 为了后面绘制每轮学习率图,此处需要记录轮次以及对应学习率
    epoch_list.append(epoch)
    lr_list.append(lr_scheduler.get_last_lr())
    # 循环迭代10次
    for i in range(10):
        # todo 预测结果
        y_pred = w * x
        # 定义损失函数
        loss = (y_pred - y_true) ** 2 / 2.0
        # 梯度清零!!!
        optimizer.zero_grad()
        # 反向传播:梯度计算
        loss.backward()
        # 参数更新
        optimizer.step()  # 底层就是w1 = w0-lr*梯度
    # TODO 每轮结束,更新学习率
    lr_scheduler.step()
# 查看最终轮次对应的学习率
print(epoch_list)
print(lr_list)

# TODO 画图 轮次对应的学习率
plt.plot(epoch_list, lr_list, label='每轮学习率')
plt.xlabel('轮次')
plt.ylabel('学习率')
plt.legend()
plt.grid()
plt.show()
