import time
import os
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

from config import Config
conf = Config()

id2name = {}
with open(conf.class_datapath, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):  # class0  class 1
        label = line.strip()
        id2name[idx] = label
# print("类别映射:", id2name)
name2id = {v: k for k, v in id2name.items()}


def model2pred(Text):
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("base_url"))

    system_prompt = '''
    你是一个优秀的文本分类师，能把给定的用户query划分到正确的类目中。现在请你根据给定信息和要求，为给定用户query，从备选类目中选择最合适的类目。

    下面是“参考案例”即被标注的正确结果，可供参考：
    文本：中国国家乒乓球队击败日本
    类别：sports

    备选类目：
    finance,realty,stocks,education,science,society,politics,sports,game,entertainment

    请注意：
    1. 用户query所选类目，仅能在【备选类目】中进行选择，用户query仅属于一个类目。
    2. “参考案例”中的内容可供推理分析，可以仿照案例来分析用户query的所选类目。
    3. 请仔细比对【备选类目】的概念和用户query的差异。
    4. 如果用户query也不属于【备选类目】中给定的类目，或者比较模糊，请选择“拒识”。
    5. 请在“文本类别：”后回复结果，不需要说明理由。

    类别:
    '''
    start_time = time.time()

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": Text},
        ],
        stream=False
    )

    result = response.choices[0].message.content
    elapsed_time = (time.time() - start_time) * 1000

    return result, elapsed_time


# 评估报告
def model2dev(dev_data):
    # 1. 初始化列表，存储预测结果和真实标签
    preds, true_labels = [], []

    start_time = time.time()

    # 2. 模型预测
    with open(dev_data, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="数据预测中....."):
            # 2.1 提取批次数据并移动到设备
            text, label = line.strip().split('\t')

            # 2.2 前向传播：模型预测
            result, _ = model2pred(text)
            pred = name2id[result]

            # 2.3 存储预测和真实标签
            preds.append(int(pred))
            true_labels.append(int(label))

    # 3. 计算分类报告、F1 分数、准确度和精确度
    accuracy = accuracy_score(true_labels, preds)  # 计算准确度
    precision = precision_score(true_labels, preds, average='micro')  # 使用微平均计算精确度
    f1score = f1_score(true_labels, preds, average='micro')  # 使用微平均计算 F1 分数
    report = classification_report(true_labels, preds)
    elapsed_time = (time.time() - start_time) * 1000
    print('预测完成！')

    # 4. 返回评估结果
    return accuracy, precision, recall_score, f1score, report, elapsed_time


if __name__ == '__main__':
    result, elapsed_time = model2pred("今日大A净流入520亿，全市超3500家上涨。")
    print(f'*预测类别：{result}')
    print(f'*请求耗时：{elapsed_time:.2f}ms')

    # accuracy, precision, recall_score, f1score, report, elapsed_time = model2dev(conf.dev_datapath)
    # print(f"*准确度:{accuracy}")
    # print(f"*精确度:{precision}")
    # print(f"*F1分数:{f1score}")
    # print(f"*分类报告:{report}")
    # print(f"*消耗时间:{elapsed_time:.2f}ms")
