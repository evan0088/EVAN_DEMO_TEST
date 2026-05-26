# 导包
import torch

# TODO 0.准备数据
# 准备真实值
y_true = torch.tensor([2, 2, 2], dtype=torch.float32)
# 准备预测值
y_pred = torch.tensor([1.2, 1.7, 1.9], requires_grad=True)
print('-----------------1.MAE损失函数-------------------------------')
# TODO 1.MAE损失函数
# 创建MAE损失函数
mae_loss_func = torch.nn.L1Loss()
# 计算MAE损失
mae_loss = mae_loss_func(y_pred, y_true)
print('MAE损失:', mae_loss)
print('----------------2.MSE损失函数----------------------------')
# TODO 2.MSE损失函数
# 创建MSE损失函数
mse_loss_func = torch.nn.MSELoss()
# 计算MSE损失
mse_loss = mse_loss_func(y_pred, y_true)
print('MSE损失:', mse_loss)
print('-----------------3.SmoothL1损失函数---------------------------')
# TODO 3.SmoothL1损失函数
# 创建SmoothL1损失函数
sl1_loss_func = torch.nn.SmoothL1Loss()
# 计算SmoothL1损失
sl1_loss = sl1_loss_func(y_pred, y_true)
print('SmoothL1损失:', sl1_loss)
