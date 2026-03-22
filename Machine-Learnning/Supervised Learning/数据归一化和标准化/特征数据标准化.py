# 1.导入归一化模型
from sklearn.preprocessing import StandardScaler

# 2.准备要处理的特征数据
x_train = [[90, 2, 10, 40],
           [60, 4, 15, 45],
           [75, 3, 13, 46]]
# 3.创建归一化模型
model = StandardScaler()
# 4.模型训练并转换数据
new_x_train = model.fit_transform(x_train)
print(f"标准化后的数据为:{new_x_train}")
"""
结果为: 
[[ 1.22474487 -1.22474487 -1.29777137 -1.3970014 ]
 [-1.22474487  1.22474487  1.13554995  0.50800051]
 [ 0.          0.          0.16222142  0.88900089]]
"""

"""
注意: 
    训练阶段：用fit_transform学习并转换特征数据。
    测试阶段：用transform严格复用训练集的规则转换特征数据。
"""