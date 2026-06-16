import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子保证可重复性
np.random.seed(42)

# 生成正样本数据 - 分布在圆周附近
n_pos = 200
theta = np.random.uniform(0, 2*np.pi, n_pos)
radius = 1.0
x_pos = radius * np.cos(theta) + np.random.normal(0, 0.1, n_pos)
y_pos = radius * np.sin(theta) + np.random.normal(0, 0.1, n_pos)
z_pos = np.random.normal(1.0, 0.1, n_pos)  # 在z轴上与负样本分离

# 生成负样本数据 - 分布在圆心附近
n_neg = 100
x_neg = np.random.normal(0, 0.2, n_neg)
y_neg = np.random.normal(0, 0.2, n_neg)
z_neg = np.random.normal(-1.0, 0.1, n_neg)  # 在z轴上与正样本分离

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制正负样本点
ax.scatter(x_pos, y_pos, z_pos, c='b', marker='o', label='正样本', alpha=0.6)
ax.scatter(x_neg, y_neg, z_neg, c='r', marker='^', label='负样本', alpha=0.6)

# 设置图形属性
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')
ax.set_title('3D散点图：正负样本分布')
ax.legend()

# 调整视角
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()
