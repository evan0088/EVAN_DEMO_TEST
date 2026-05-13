from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, Qwen2Tokenizer
import torch


# 具体模型完型填空任务
def dm01_test_bert_fill_mask():
    # 1 加载tokenizer
    modename = "model/bert-base-chinese"
    my_tokenizer = BertTokenizer.from_pretrained(modename)

    # 2 加载模型
    my_model = BertForMaskedLM.from_pretrained(modename)

    # 3 文本转张量
    input = my_tokenizer.encode_plus('我想明天去[MASK]家吃饭', return_tensors='pt')
    print('input--->', input)

    # 4 给模型送数据提取特征
    my_model.eval()
    output = my_model(**input)
    print('output--->', output)
    print('output.logits--->', output.logits.shape) # [1,11,21128]

    # 5 取概率最高
    mask_pred_idx = torch.argmax(output.logits[0][6]).item()
    print('打印概率最高的字:', my_tokenizer.convert_ids_to_tokens([mask_pred_idx]))


if __name__ == '__main__':
    dm01_test_bert_fill_mask()