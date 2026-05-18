# 导入必要的库
import pandas as pd
import jieba
from config import Config

# 初始化配置文件
conf = Config()
#设置处理的及分析的文件路径，默认为 train.txt
current_path=conf.train_datapath
# current_path=conf.test_datapath
# current_path=conf.dev_datapath

# 第一步：读取数据
df = pd.read_csv(current_path, sep='\t', names=['text', 'label'])  # 读取CSV文件，指定列名为text和label

# 第二步：进行分词预处理
def cut_sentence(s):
    """对输入文本进行结巴分词，并限制前30个词"""
    return ' '.join(list(jieba.cut(s))[:30])  # 分词后取前30个词并a用空格连接
df['words'] = df['text'].apply(cut_sentence)  # 对每行文本进行分词并存储到words列
print("*"*60)
print(df.head(10))

# 第三步：保存处理后的数据
if "train" in current_path:
    df.to_csv(conf.processt_train_datapath, index=False)# 将处理后的数据保存到CSV文件
    print(f"train数据已经处理完成，已经成功保存至：{conf.processt_train_datapath}")
elif "test" in current_path:
    df.to_csv(conf.proces_test_datapath, index=False)# 将处理后的数据保存到CSV文件
    print(f"test数据已经处理完成，已经成功保存至：{conf.proces_test_datapath}")
elif "dev" in current_path:
    df.to_csv(conf.proces_dev_datapath, index=False)# 将处理后的数据保存到CSV文件
    print(f"dev数据已经处理完成，已经成功保存至：{conf.proces_dev_datapath}")
