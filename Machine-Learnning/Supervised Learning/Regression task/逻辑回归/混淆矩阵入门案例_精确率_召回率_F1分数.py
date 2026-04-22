# 1.导包
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report
#
# 2.TODO 需求: 已知10个样本,6个恶性,4个良性,根据AB模型预测结果绘制混淆矩阵
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
print('==================================================================')
# todo B模型预测并创建混淆矩阵
# B预测对了6个恶性,1个良性
y_pred_B = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", '恶性', '恶性', '恶性', '良性']
# 此处labels可以省略,默认恶性是正例
cm_B = confusion_matrix(y_test, y_pred_B, labels=["恶性", "良性"])
print('B模型混淆矩阵:\n', cm_B)
print('==================================================================')
# 3.TODO 模型评估常用: 准确率,精确率,召回率,F1分数
print('A模型准确率:', accuracy_score(y_test, y_pred_A))
print('A模型精确率:', precision_score(y_test, y_pred_A,pos_label="恶性"))
print('A模型召回率:', recall_score(y_test, y_pred_A,pos_label="恶性"))
print('A模型F1分数:', f1_score(y_test, y_pred_A,pos_label="恶性"))
print('==================================================================')
print('B模型准确率:', accuracy_score(y_test, y_pred_B))
print('B模型精确率:', precision_score(y_test, y_pred_B,pos_label="恶性"))
print('B模型召回率:', recall_score(y_test, y_pred_B,pos_label="恶性"))
print('B模型F1分数:', f1_score(y_test, y_pred_B,pos_label="恶性"))
print('==================================================================')
# 4.TODO 模型评估拓展: 分类报表直接打印
print('分类报表:',classification_report(y_test, y_pred_A))
print('分类报表:',classification_report(y_test, y_pred_B))