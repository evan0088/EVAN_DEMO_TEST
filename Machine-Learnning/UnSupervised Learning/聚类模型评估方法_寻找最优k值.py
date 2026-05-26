# 1.导包
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2.准备散点数据集
x, y = make_blobs(
    n_samples=1000,  # 默认100
    n_features=2,
    centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],  # 效果看默认3
    cluster_std=[0.4, 0.2, 0.2, 0.3],  # 默认1
)
# TODO 3.循环查找最优k值
sse_list = []
for k in range(1, 10):
    # 创建KMeans模型
    # 4.kmeans算法API应用
    model = KMeans(n_clusters=k)
    y_pre = model.fit_predict(x)
    # print(y_pre)
    # TODO 采用SSE肘方法(肘峰法则)肘峰位置就是最优k值
    value = model.inertia_
    sse_list.append(value)
# 5.展示聚类后数据的散点图
fig = plt.figure(figsize=(10,10))
fig.add_subplot(111) # 1个1行1列的表格
plt.plot(range(1, 10), sse_list)
plt.title("肘方法")
plt.xlabel("k值")
plt.ylabel("SSE结果")
plt.show()
