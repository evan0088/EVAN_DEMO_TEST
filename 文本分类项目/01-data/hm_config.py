#定义配置文件类
#加载文件路径 train_datapath、test_datapath、dev_datapath、class_datapath
class Config():
    def __init__(self):
        self.train_datapath="./train.txt"
        self.test_datapath="./test.txt"
        self.dev_datapath="./dev.txt"
        self.class_datapath="./class.txt"
        self.demo = '测试专业'


if __name__ == '__main__':
    config = Config()
    print(f'测试集路径：{config.test_datapath}')
    print(f'类别文件路径：{config.class_datapath}')