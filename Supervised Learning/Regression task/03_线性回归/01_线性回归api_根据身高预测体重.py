# 1.导包
from sklearn.linear_model import LinearRegression

# 2.准备数据
x_train = [[160], [166], [172], [174], [180]]
y_train = [56.3, 60.6, 65.1, 68.5, 75]
x_test = [[176]]
# 3.创建线性模型
model = LinearRegression()
# 4.模型训练
# todo 训练目的: 找到先对最优的斜率和截距
model.fit(x_train, y_train)
# todo 打印训练结果
print(f"最终的斜率:{model.coef_}")  # [0.92942177]
print(f"最终的截距:{model.intercept_}")  # -93.27346938775517
# 5.模型预测
# todo 方式1: 手动套入公式
print(f"手动计算结果:{model.coef_[0] * 176 + model.intercept_}")
# todo 方式2: 使用模型自带的预测方法
y_test = model.predict(x_test)
print(f"模型预测结果:{y_test}")
