
# 注意: 导包(todo 直接报错,因为这是老版本的,报错信息中告诉你新版本方式,直接复制即可)
# from sklearn.datasets import load_boston
# 赋值新版本数据如下
import pandas as pd
import numpy as np

# todo 1.获取数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data.shape)  # (506, 13)
print(target.shape)  # (506,)

# todo 2.数据切割
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)
# todo 3.数据标准化处理
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
new_x_train = ss.fit_transform(x_train)
new_x_test = ss.transform(x_test)
# TODO  4.创建模型(正规方程)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# todo 5.模型训练
model.fit(new_x_train, y_train)
print(f"训练后coef_参数:{model.coef_}")
print(f"训练后intercept_参数:{model.intercept_}")
# todo 6.模型预测
y_pred = model.predict(new_x_test)
# todo 7.模型评估
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
print(f"均方误差:{mean_squared_error(y_test, y_pred)}")
print(f"均方根误差:{root_mean_squared_error(y_test,y_pred)}")
print(f"平均绝对误差:{mean_absolute_error(y_test, y_pred)}")

