# 1.导入knn分类模型
from sklearn.neighbors import KNeighborsRegressor

# 2.准备数据集(三缺一)
x_train = [[0], [1], [2], [3]]
x_test = [[4]]
# 标签是房价,比如70万, 80万, 100万, 120万
y_train = [70, 80, 100, 120]
# 3.创建knn分类模型
# 按住ctrl+鼠标左键点击查看源码
model = KNeighborsRegressor(n_neighbors=3)
# 4.模型训练
model.fit(x_train, y_train)
# 5.模型预测
y_test = model.predict(x_test)
print(f"分类预测结果为:{y_test}")
