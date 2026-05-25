
import torch
import os
import datetime
from transformers.models import BertModel,BertTokenizer,BertConfig
current_date=datetime.datetime.now().date().strftime("%Y%m%d")

class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert" # 模型名称
        self.data_path = "../../01-data"  #数据集的根路径
        self.train_path = self.data_path + "\\train.txt"  # 训练集
        self.dev_path = self.data_path + "\\dev3.txt"  # 少量验证集，快速验证
        # self.dev_path = self.data_path + "\\dev.txt"  # 全量验证集
        self.test_path = self.data_path + "\\test.txt"  # 测试集
        self.class_num = len([line.strip() for line in open("../../01-data/class.txt",encoding="utf-8")]) # 类别名单

        # BERT原模型训练结果保存路径
        self.model_save_dir = "../../04-bert/save_models"
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        self.model_save_path = self.model_save_dir +"\\bert20250521_.pt"

        # BERT蒸馏模型存储结果路径
        self.distil_model_save_dir = "./models_save"
        if not os.path.exists(self.distil_model_save_dir):
            os.mkdir(self.distil_model_save_dir)
        self.distil_h_model_save_path = self.distil_model_save_dir + "\\student_distll_h.pt"
        self.distil_s_model_save_path = self.distil_model_save_dir + "\\student_distll_s.pt"

        # 模型训练+预测的时候, 放开下一行代码, 在GPU上运行.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 1  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "../../04-bert/bert-base-chinese"  # 预训练BERT模型的路径
        self.bert_model=BertModel.from_pretrained(self.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path) # BERT模型的配置
        self.hidden_size = 768 # BERT模型的隐藏层大小

        # BiLSTM模型参数配置
        # self.embed_size = 128
        # self.hidden_size_lstm = 256
        # self.num_layers = 2
        # self.dropout = 0.3
        # self.save_model_path = "./models_save"
        # self.dropout=0.3

        self.embed_size = 256
        self.hidden_size_lstm = 512
        self.num_layers = 4
        self.dropout = 0.3
        self.save_model_path = "./models_save"
        self.dropout=0.3

if __name__ == '__main__':
    conf = Config()
    print(conf.bert_config)
    input_size=conf.tokenizer.convert_tokens_to_ids(["你", "好", "中国人"])
    print(input_size)