# 导包
import torch

# 手动准备真实值
# todo 方式1: 使用热编码方式(推荐)  label
y_true = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float)
# 方式2: 使用正确值索引方式(了解)  1             2
#                         [[0, 1, 0], [0, 0, 1]]
# y_true = torch.tensor([1, 2],  dtype=torch.int64)
# todo 手动准备预测值(开启自动微分)  logits
y_pred = torch.tensor([[0.1, 0.7, 0.2], [0.1, 0.3, 0.6]], requires_grad=True, dtype=torch.float32)
# TODO 计算损失
# 创建损失函数对象
loss_func = torch.nn.CrossEntropyLoss()
#  计算损失
loss = loss_func(y_pred, y_true)
#  打印损失
print(loss)
