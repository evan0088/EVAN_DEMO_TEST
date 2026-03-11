#!/usr/bin/env python
# @desc : 
__coding__ = "utf-8"
__author__ = "itcast team"

from sklearn.neighbors import KNeighborsRegressor

x_train = [[0], [1], [2], [3]]
y_train = [70, 100, 120, 80]
x_test = [[4]]

model = KNeighborsRegressor(n_neighbors=3)

model.fit(x_train, y_train)
y_test = model.predict(x_test)
print(y_test)