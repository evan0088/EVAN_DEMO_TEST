# 导入需要的工具
import pandas as pd  # 用于处理表格数据
from collections import Counter  # 用于统计标签
from hm_config import Config  # 导入配置类

# 创建配置对象，获取数据文件路径
config = Config()
# file_path = config.train_datapath  # 默认使用训练数据文件 train.txt
file_path = "dev3.txt"

# 第一步：读取数据并查看基本信息
data = pd.read_csv(file_path, sep='\t', names=['text', 'label'])  # 读取文件，列名为“文本”和“标签”
print(f'*****前5行数据*****\n{data.head(5)}')  # 显示前5行，了解数据长什么样
print(f"\n*****总数据量*****\n{len(data)} 行")  # 显示总行数

# 第二步：统计标签分布
label_counts = Counter(data['label'])  # 数一数每个标签出现了几次
print("\n*****标签分布*****")
for label, count in label_counts.items():
    print(f"标签 {label}：{count} 次")  # 输出每个标签的次数

# 第三步：计算标签比例
total_rows = len(data)  # 总行数
print("\n*****标签比例*****")
for label, count in label_counts.items():
    percent = (count / total_rows) * 100  # 计算百分比
    print(f"标签 {label}：{percent:.2f}%")  # 输出百分比，保留2位小数

# 第四步：分析文本长度
data['text_length'] = data['text'].str.len()  # 计算每条文本的字符数
print(f"\n*****文本长度前10行*****\n:{data[['text', 'text_length']].head(10)}")  # 只显示文本和长度列
print("\n*****文本长度统计*****")
print(f"平均长度：{data['text_length'].mean():.2f} 字符")  # 平均值
print(f"长度标准差：{data['text_length'].std():.2f} 字符")  # 标准差
print(f"最大长度：{data['text_length'].max()} 字符")  # 最大值
print(f"最小长度：{data['text_length'].min()} 字符")  # 最小值
