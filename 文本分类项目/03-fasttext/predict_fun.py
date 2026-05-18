# 模型预测
import fasttext
import jieba
import warnings
warnings.filterwarnings('ignore')
#1、加载模型
model = fasttext.load_model('save_models/fastText_char_default_20250630.bin')
#2、定义预测函数
def predict(data):
    #获取text进行进行字符级别分词
    words=" ".join(list(data["text"]))
    #模型预测
    res=model.predict(words)
    #截取 预测返回值
    pred_label=res[0][0][9:]
    #封装结果并返回
    data["pred_class"]=pred_label
    return data

data = {"text": "中华女子学院：本科层次仅1专业招男生"}
print(predict(data))