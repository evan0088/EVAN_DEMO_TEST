import os
class Config(object):
    def __init__(self):
        #原始数据路径
        self.train_datapath="../01-data/train.txt"
        self.test_datapath = "../01-data/test.txt"
        self.dev_datapath = "../01-data/dev.txt"

        #模型路径
        self.ft_model_save_path= "save_models"

        #样本类别文件
        self.class_datapath="../01-data/class.txt"

        #处理完的数据（用于训练）
        self.final_data="./final_data"

        # 是否使用字符级别的分割
        self.use_char_segmentation = False

if __name__ == '__main__':
    conf=Config()
    print(conf.ft_model_save_path)
    print(conf.train_datapath)