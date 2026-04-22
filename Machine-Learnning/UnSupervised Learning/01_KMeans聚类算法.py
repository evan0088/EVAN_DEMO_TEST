"""
K-Means聚类算法 - 简单案例
无监督学习:不需要标签,自动将数据分成K个簇
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端,避免兼容性问题
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据 - 模拟客户消费数据(年收入和消费分数)
np.random.seed(42)
data = np.array([
    # 低收入低消费群体
    [20, 20], [25, 25], [30, 30], [22, 18], [28, 22],
    # 中等收入中等消费群体
    [50, 50], [55, 45], [60, 55], [48, 52], [52, 48],
    # 高收入高消费群体
    [80, 80], [85, 75], [90, 85], [78, 82], [88, 78]
])

print("原始数据:")
print(data)

# 2. 创建K-Means模型,设置聚成3类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# 3. 训练模型(拟合数据)
kmeans.fit(data)

# 4. 获取每个数据点的簇标签
labels = kmeans.labels_
print("\n每个数据点的簇标签:")
print(labels)

# 5. 获取簇中心点
centers = kmeans.cluster_centers_
print("\n簇中心点:")
print(centers)

# 6. 可视化结果
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=100)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='簇中心')
plt.xlabel('年收入 (千元)')
plt.ylabel('消费分数')
plt.title('K-Means聚类结果 - 客户分群')
plt.legend()
plt.colorbar(scatter, label='簇类别')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 先保存图片(使用Agg后端不需要show)
output_path = 'kmeans_clustering.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n聚类完成!图片已保存为 {output_path}")
