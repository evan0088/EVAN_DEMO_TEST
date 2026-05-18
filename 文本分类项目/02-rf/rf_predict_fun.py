import jieba
import pandas as pd
import pickle
from config import Config
import warnings

warnings.filterwarnings('ignore')
# 设置pandas显示选项
pd.set_option('display.max_columns', None)


def predict(data):
    # 加载配置
    conf = Config()
    # 第一步：加载保存的模型和向量化器
    with open(conf.rf_model_save_path + '/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(conf.rf_model_save_path + '/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    #  第二步：对输入数据进行切分
    words = " ".join(jieba.lcut(data['text'])[:30])  #####

    #  第三步：对输入数据进行向量化
    feature = tfidf.transform([words])

    #  第四步：对输入数据进行预测
    y_pred = model.predict(feature)
    # 第五步：转换预测结果，并进行返回预测结果
    id2class = {index: label.strip() for index, label in enumerate(open(conf.class_datapath))}
    data["pred_class"] = id2class[y_pred[0]]
    print(data)
    return data


data = {"text": "体验2D巅峰 倚天屠龙记十大创新概览"}
print(predict(data))
