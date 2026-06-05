import json
import torch
from .common import *
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

"""
需求：基于预处理数据(x,y)样本对、词表等数据，构建数据迭代器
思路步骤：
1. 构建NerDataset类
	1.1 继承Dataset，实现__init__，获取预处理数据( (x,y)样本对数据集  )
	1.2.基于深度学习实现NER 实现__len__：返回数据集大小
	1.3 实现__getitem__: 返回item索引位置对应的(x,y)样本对
2.基于深度学习实现NER. 构建自定义函数collate_fn()
	2.基于深度学习实现NER.1 对(x,y)样本对进行编码，进行word2id和tag2id的转换
	2.基于深度学习实现NER.2.基于深度学习实现NER 对x进行长度补齐，得到特征输入
	2.基于深度学习实现NER.3 对y进行长度补齐，得到标签输入
	2.基于深度学习实现NER.4 对x计算注意力掩码，得到掩码矩阵
3. 构建get_data函数，获得数据迭代器
	3.1 对数据集进行训练集/测试集分割
	3.2.基于深度学习实现NER 基于NerDataset、collate_fn、和配置信息和构建Dataloader
"""

datas, word2id = build_data()


class NerDataset(Dataset):
    def __init__(self, datas):
        super().__init__()
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        x = self.datas[item][0]
        y = self.datas[item][1]
        return x, y


def collate_fn(batch):
    # TODO : 核心处理逻辑（X和Y都是一样的）
    # 1. for data in batch： 拿到每一个（x,y）样本对
    # 2.基于深度学习实现NER. for char in data[0] ：拿到每一个x中的所有的token
    # 3. word2id[char] ：对每一个token进行转换，转成token_id
    # 4. 把每个句子的tokenid转成tensor (一维的张量， 每个张量代表的一个句子)
    # 5. 返回list格式的结果，不能加tensor。 因为后面pad_sequence的方法需要接收list数据

    x_train = [torch.tensor([word2id[char] for char in data[0]]) for data in batch]
    y_train = [torch.tensor([conf.tag2id[label] for label in data[1]]) for data in batch]
    # 补齐input_ids, 使用0作为填充值
    input_ids_padded = pad_sequence(x_train, batch_first=True, padding_value=0)
    # # 补齐labels，使用0作为填充值
    # labels_padded = pad_sequence(y_train, batch_first=True, padding_value=-100)
    # 因为这里0的意思O（非实体），我们可以把空白当成非实体
    labels_padded = pad_sequence(y_train, batch_first=True, padding_value=0)

    # 创建attention mask
    # 作用是为了还原真实的长度
    attention_mask = (input_ids_padded != 0).long()
    return input_ids_padded, labels_padded, attention_mask


def get_data():
    # :6200：切片操作是为了划分数据集， 总共大概7800条
    train_dataset = NerDataset(datas[:6200])
    train_dataloader = DataLoader(
                                  dataset=train_dataset,
                                  batch_size=conf.batch_size,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  )

    dev_dataset = NerDataset(datas[6200:])
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=conf.batch_size,
                                collate_fn=collate_fn,
                                drop_last=True,
                                )
    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    for input_ids_padded, labels_padded, attention_mask in train_dataloader:
        print(f'input_ids_padded --> {input_ids_padded}')
        print(f'labels_padded --> {labels_padded}')
        print(f'attention_mask --> {attention_mask}')
        break
        # print(input_ids_padded.shape)
        # print(labels_padded.shape)
        # print(attention_mask.shape)
        # print('*' * 100)
