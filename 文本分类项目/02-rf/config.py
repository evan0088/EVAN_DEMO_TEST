class Config(object):
    def __init__(self):
        # 原始数据路径
        self.train_datapath = "../01-data/train.txt"
        self.test_datapath = "../01-data/test.txt"
        self.dev_datapath = "../01-data/dev.txt"
        self.class_datapath = "../01-data/class.txt"

        # 处理后的数据路径
        self.processt_train_datapath = "./data/process_train.csv"
        self.proces_test_datapath = "./data/process_test.csv"
        self.proces_dev_datapath = "./data/process_dev.csv"

        # 停用词路径
        self.stop_words_path = "../01-data/stopwords.txt"

        # 保存模型路径
        self.rf_model_save_path = "save_model"
        self.model_predict_result = "./result"
        # self.WERKZEUG_RUN_MAIN=True


if __name__ == '__main__':
    conf = Config()
    print(conf.train_datapath)
    print(conf.test_datapath)
