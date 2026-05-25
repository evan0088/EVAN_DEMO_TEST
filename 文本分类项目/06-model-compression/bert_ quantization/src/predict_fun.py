import time

import torch
from transformers import BertTokenizer
from bert_classifer_model import BertClassifier
import warnings
warnings.filterwarnings("ignore")

from config import Config
conf = Config()

model = torch.load(conf.quantized_model_save_path, map_location=conf.device,weights_only=False)

def predict(data):
    # 1.处理输入数据data["text"]
    text = data["text"]
    if not text.strip():
        return {"text": text, "pred_class": None}

    # 2.分词并编码 tokenizer.encode_plus,支持返回pt
    encoded = conf.tokenizer.encode_plus(text, return_tensors="pt")
    # 获取input_ids与 attention_mask
    input_ids = encoded["input_ids"].to(conf.device)
    attention_mask = encoded["attention_mask"].to(conf.device)
    # 3.模型预测
    # 3.1 关闭梯度计算  with torch.no_grad():
    with torch.no_grad():
        start_time = time.time()
        # 3.2 前向传播
        logits = model(input_ids, attention_mask)
        # 3.3 获取预测结果，torch.argmax获取最大logits的索引pred_idx
        pred_idx = torch.argmax(logits, dim=1).item()
        # 获取预测的类别conf.class_list[pred_idx]
        pred_class = conf.class_list[pred_idx]
        elaspe_time = (time.time() - start_time) * 1000

    return {"text": text, "pred_class": pred_class, "elaspe_time": elaspe_time}


if __name__ == "__main__":
    # 1.初始化配置
    conf = Config()
    # 2.获取测试输入
    sample_data = {"text": "中华女子学院：本科层次仅1专业招男生"}
    # 3.实例化模型
    model = torch.load(conf.quantized_model_save_path, map_location=conf.device,weights_only=False)
    # 4.模型预测
    result = predict(sample_data)
    print(f"预测文本：{result['text']}")
    print(f"预测结果：{result['pred_class']}")
    print(f"预测耗时：{result['elaspe_time']:.2f}ms")
