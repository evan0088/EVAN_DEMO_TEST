# 1.导包
from sklearn.metrics import confusion_matrix
import pandas as pd
# 2.准备数据集
# TODO 需求: 已知10个样本,6个恶性,4个良性,根据AB模型预测结果绘制混淆矩阵
# 人工真实标签
y_test = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", '良性', '良性', '良性', '良性']
print('人工真实标签:', y_test)
print('==================================================================')
# todo A模型预测并创建混淆矩阵
# A预测对了3个恶性,4个良性
y_pred_A = ["恶性", "恶性", "恶性", "良性", "良性", "良性", '良性', '良性', '良性', '良性']
# 此处labels可以省略,默认恶性是正例
cm_A = confusion_matrix(y_test, y_pred_A, labels=["恶性", "良性"])
print('A模型混淆矩阵:\n', cm_A)
# TODO pandas友好展示混淆矩阵
cm_A_df = pd.DataFrame(cm_A, index=["恶性(正例)", "良性(反例)"], columns=["恶性(正例)", "良性(反例)"])
print('A模型混淆矩阵df格式:\n', cm_A_df)
print('==================================================================')
# todo B模型预测并创建混淆矩阵
# B预测对了6个恶性,1个良性
y_pred_B = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", '恶性', '恶性', '恶性', '良性']
# 此处labels可以省略,默认恶性是正例
cm_B = confusion_matrix(y_test, y_pred_B, labels=["恶性", "良性"])
print('B模型混淆矩阵:\n', cm_B)
# TODO pandas友好展示混淆矩阵
cm_B_df = pd.DataFrame(cm_B, index=["恶性(正例)", "良性(反例)"], columns=["恶性(正例)", "良性(反例)"])
print('A模型混淆矩阵df格式:\n', cm_B_df)