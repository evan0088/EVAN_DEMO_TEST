# todo 1.加载数据
import pandas as pd
import numpy as np

data = df = pd.read_csv("data/breast-cancer-wisconsin.csv")
print(data.shape)  # (699, 11)
# todo 2.数据预处理
# 需求: 数据中有?的数据,需要处理(先替换成NaN,然后删除)
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
print(data.shape)  # (683, 11)
# todo 3.获取特征值和目标值
# 拓展: iloc格式: [行切片,列切片]
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
# todo 4.切割数据
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# todo 5.数据标准化处理
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
new_x_train = ss.fit_transform(x_train)
new_x_test = ss.transform(x_test)
# TODO 6.创建逻辑回归模型
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# todo 7.模型训练
model.fit(new_x_train, y_train)
# todo 8.模型预测
y_pred = model.predict(new_x_test)
# todo 计算准确率
# todo 方式1: 打印出来自己数,自己算
print(y_test.tolist())
print(y_pred)
# todo 方式2: accuracy_score
from sklearn.metrics import accuracy_score

print(f"准确率:{accuracy_score(y_test, y_pred)}")
# todo 方式3: score
print(f"准确率:{model.score(new_x_test, y_test)}")
