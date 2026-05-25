class Config(object):
    def __init__(self):
        # 原始数据路径
        self.train_datapath = "../01-data/train.txt"
        self.test_datapath = "../01-data/test.txt"
        self.dev_datapath = "../01-data/dev.txt"
        self.class_datapath = "../01-data/class.txt"


if __name__ == '__main__':
    conf = Config()
    print(conf.train_datapath)
    print(conf.test_datapath)
