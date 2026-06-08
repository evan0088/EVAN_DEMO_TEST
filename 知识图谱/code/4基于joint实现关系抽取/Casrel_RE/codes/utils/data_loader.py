"""
需求：准备训练、验证和测试数据集的DataLoader对象
思路步骤：
    1. 导入必要的库和配置
    2. 定义自定义的数据集类MyDataset：
        2.1 在构造函数中，读取指定路径的数据文件并解析为JSON格式存储
        2.2 实现__len__方法，返回数据集的长度
        2.3 实现__getitem__方法，获取指定索引的数据项，返回文本和spo列表
    3. 定义get_data函数：
        3.1 实例化训练、验证和测试数据集的MyDataset对象
        3.2 分别实例化训练、验证和测试数据集的DataLoader对象，设置批量大小、打乱顺序、整理函数等参数
        3.3 返回训练、验证和测试数据集的DataLoader对象
"""
# coding:utf-8
from torch.utils.data import DataLoader, Dataset
from codes.utils.process import *

conf = Config()


# 自定义Dataset
class MyDataset(Dataset):
    # 在构造函数中，读取指定路径的数据文件并解析为JSON格式存储
    def __init__(self, data_path):
        super(MyDataset, self).__init__()
        self.dataset = [json.loads(line) for line in open(data_path, encoding='utf8')]

    # 实现__len__方法，返回数据集的长度
    def __len__(self):
        return len(self.dataset)

    # 实现__getitem__方法，获取指定索引的数据项，返回文本和spo列表
    def __getitem__(self, index):
        content = self.dataset[index]
        text = content['text']
        spo_list = content['spo_list']
        return text, spo_list


def get_data():
    # 实例化训练数据集Dataset对象
    train_data = MyDataset(conf.train_data_path)

    # 实例化验证数据集Dataset对象
    dev_data = MyDataset(conf.dev_data_path)

    # 实例化测试数据集Dataset对象
    test_data = MyDataset(conf.test_data_path)

    # 实例化训练数据集Dataloader对象
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=conf.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    # 实例化验证数据集Dataloader对象
    dev_dataloader = DataLoader(dataset=dev_data,
                                batch_size=conf.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                drop_last=True)
    # 实例化测试数据集Dataloader对象
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=conf.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True)
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader, test_dataloader = get_data()

    for inputs, labels in train_dataloader:
        for input in inputs:
            print(f'{input} --> {inputs[input].shape}')

        for label in labels:
            print(f'{label} --> {labels[label].shape}')
        print("*" * 100)
