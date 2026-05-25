import time
import torch
from bilstm_classifier import BiLSTMClassifier
from config import Config

class_list = [line.strip() for line in open("../../01-data/class.txt", encoding="utf-8")]

# 预测函数
def predict(data):
    # 处理输入数据 data["text"]
    text = data["text"]
    if not text.strip():
        return {"text": text, "pred_class": None}

    # 分词并编码，使用 tokenizer.encode_plus，返回 PyTorch 张量
    encoded = conf.tokenizer.encode_plus(text, return_tensors="pt")
    # 获取 input_ids 和 attention_mask
    input_ids = encoded["input_ids"].to(conf.device)
    attention_mask = encoded["attention_mask"].to(conf.device)

    # 开启模型推理模式
    with torch.no_grad():
        # 开始时间
        start_time = time.time()
        # 模型预测
        logits = model(input_ids, attention_mask)
        # 获取最大 logits 的索引
        pred_idx = torch.argmax(logits, dim=1).item()
        # 获取预测的类别
        pred_class = class_list[pred_idx]
        # 预测时间
        elaspe_time = (time.time()-start_time)*1000

    return {"text": text, "pred_class": pred_class, "elaspe_time": elaspe_time}

if __name__ == "__main__":
    # 1.初始化配置
    conf = Config()
    # 2.实例化 BiLSTM 模型
    model = BiLSTMClassifier(conf).to(conf.device)
    # 3.加载预训练模型权重（需替换为实际路径）
    model.load_state_dict(torch.load(conf.distil_s_model_save_path)) # 软标签蒸馏模型
    # model.load_state_dict(torch.load(conf.distil_h_model_save_path)) # 硬标签蒸馏模型
    model.eval()
    # 测试输入
    sample_data = {"text": "中华女子学院：本科层次仅1专业招男生"}
    result = predict(sample_data)
    print(f"预测文本：{result['text']}")
    print(f"预测结果：{result['pred_class']}")
    print(f"预测耗时：{result['elaspe_time']:.2f}ms")