# 导入工具包
import fasttext
from config import Config
import datetime
import random
import numpy as np
import os

# 获取当前日期
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")

# 1. 导入配置文件
conf = Config()

# 2. 模型训练
model = fasttext.train_supervised(
    input='./final_data/train_fastText_char.txt',
    autotuneValidationFile='./final_data/dev_fastText_char.txt',
    autotuneDuration=60,  # 搜索的时间  默认300s
    thread=1,  # 单线程，确保可复现性
    verbose=3  # 输出调参过程
)

# 3. 模型保存
path = conf.ft_model_save_path
model_save_path = path + f"\\fastText_char_auto_{str(current_time)}.bin"
model.save_model(model_save_path)
print(f"模型已保存至: {model_save_path}")

# 4. 模型预测
sentence = "俄 达 吉 斯 坦 共 和 国 一 名 区 长 被 枪 杀"
pred_label, pred_prob = model.predict(sentence)
print(f"预测结果: 标签={pred_label[0]}, 概率={pred_prob[0]:.4f}")

# 5. 查看模型子词
word = "好"
subwords, subword_ids = model.get_subwords(word)
print(f"*词'{word}'的子词:{subwords}")
print(f"*子词ID:{subword_ids}")

# 获取词向量维度
print(f'*词向量维度:{model.get_dimension()}')

# 6. 模型测试评估
res = model.test('./final_data/test_fastText_char.txt')
# 参考官方说明：https://fasttext.cc/docs/en/supervised-tutorial.html
print(f"测试结果: 样本数={res[0]}, 精确率={res[1]:.4f}, 召回率={res[2]:.4f}")
