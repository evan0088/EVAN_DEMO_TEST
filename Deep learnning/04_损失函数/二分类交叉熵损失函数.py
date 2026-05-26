# 导包
import torch

# 准备真实值
# todo 使用热编码方式
y_true = torch.tensor([0, 1, 0], dtype=torch.float)
# 准备预测值(开启自动微分)
y_pred = torch.tensor([0.1, 0.7, 0.2], requires_grad=True, dtype=torch.float32)
# TODO 计算损失
# 创建损失函数对象
criterion = torch.nn.BCELoss()
#  计算损失
loss = criterion(y_pred, y_true)
#  打印损失
print(loss)
