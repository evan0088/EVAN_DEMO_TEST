# 导入工具包
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from rich import print
from torch.optim import AdamW
import time

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载字典和分词工具 实例化分词工具
my_tokenizer = BertTokenizer.from_pretrained('model/bert-base-chinese')

# 加载预训练模型 实例化预训练模型
my_model_pretrained = BertModel.from_pretrained('model/bert-base-chinese').to(device)
# print('my_model_pretrained--->', my_model_pretrained)

# 查看预训练模型的输出维度
hidden_size = my_model_pretrained.config.hidden_size
print('hidden_size--->', hidden_size)  # 768


# todo:1-加载数据集
def dm_file2dataset():
    # 实例化数据源对象my_dataset_train
    # print('\n加载训练集')
    # print(load_dataset('csv', data_files='data/train.csv'))
    # split='train': 获取key对应的value  key的值为train
    my_dataset_train = load_dataset('csv', data_files='data/train.csv', split='train')
    # print('dataset_train--->', my_dataset_train)
    # print(my_dataset_train[0:3])
    # print('=' * 80)

    # 实例化数据源对象my_dataset_test
    # print('\n加载测试集')
    # split='train': 获取key对应的value  key的值为train
    # print(load_dataset('csv', data_files='data/test.csv'))
    my_dataset_test = load_dataset('csv', data_files='data/test.csv', split='train')
    # print('my_dataset_test--->', my_dataset_test)
    # print(my_dataset_test[0:3])
    # print('=' * 80)

    # print('\n加载验证集')
    # 实例化数据源对象my_dataset_train
    my_dataset_validation = load_dataset('csv', data_files='data/validation.csv', split="train")
    # print('my_dataset_validation--->', my_dataset_validation)
    # print(my_dataset_validation[0:3])
    # print('=' * 80)
    return my_dataset_train, my_dataset_test, my_dataset_validation


# todo:2-构建数据加载器自定义函数
# 数据集处理自定义函数
# 必须定义一个形参
def collate_fn(data):
    """
    根据函数逻辑对批次样本进行处理
    :param data: 批次样本数据, 必传
    :return: 处理完成的批次样本数据
    """
    # print('data--->', data)
    # data传过来的数据是list eg: 批次数8，8个字典
    # [{'text':'xxxx','label':0} , {'text':'xxxx','label':1}, ...]
    # 列表推导式 i->字典类型 i['text']->通过key获取value
    # 获取批次样本的句子列表
    sents = [i['text'] for i in data]
    # print('sents--->', sents)
    # 获取批次样本的标签列表
    labels = [i['label'] for i in data]
    # print('labels--->', labels)

    # 编码text2id 对多句话进行编码用batch_encode_plus函数
    data = my_tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=500,
                                          return_tensors='pt')
    # print('data--->', data)

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    # data['input_ids']: 字典通过key获取value
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    # 返回text2id信息 掩码信息 句子分段信息 标签y
    return input_ids, attention_mask, token_type_ids, labels


# todo:3-构建下游任务的网络模型
# 定义下游任务模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义全连接层
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 预训练模型不训练 只进行特征抽取 [8,500] ---> [8,768]
        with torch.no_grad():
            # 句子的语义表示
            out = my_model_pretrained(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)

        # print('out--->', out)
        # 下游任务模型训练 数据经过全连接层 [8,768] --> [8,2]
        # out.last_hidden_state: 最后一层的隐藏状态张量
        # [:, 0] 选择的是序列的第一个token的隐藏状态
        # 通常这个token是特殊的[CLS]，该token被设计用于表示整个序列的语义。
        # BERT训练时，特别是文本分类任务，使用[CLS]的表示来作为整个句子的表示。
        # out = self.fc(out.last_hidden_state[:, 0])

        # pooler_output: 通过last_hidden_state[:, 0]拿到[CLS]向量表示后又经过linear层(形状不变)
        out = self.fc(out.pooler_output)
        # print('out--->', out)
        return out


# 测试
def dm01_test():
    # 加载数据集
    my_dataset_train, my_dataset_test, my_dataset_validation = dm_file2dataset()
    # 构建数据加载器对象
    # batch_size: 批次大小, 一批多少个句子样本
    # collate_fn: 自定义函数名, 每批样本根据自定义函数代码处理逻辑进行处理
    my_dataloader = DataLoader(dataset=my_dataset_train,
                               collate_fn=collate_fn,
                               shuffle=True,
                               batch_size=8,
                               drop_last=True)

    # 冻结预训练模型 embedding层参数
    # for param in my_model_pretrained.embeddings.parameters():
    #     param.requires_grad_(False)

    # 冻结预训练模型的前10层encoder层参数
    # for i, layer in enumerate(my_model_pretrained.encoder.layer):
    #     if i < 10:
    #         for param in layer.parameters():
    #             param.requires_grad = False

    # 不训练,不需要计算梯度 双保险
    # for param in my_model_pretrained.parameters():
    #     param.requires_grad_(False)

    # 实例化下游任务模型
    my_model = MyModel().to(device)
    print('my_model--->', my_model)

    # 调整数据迭代器对象数据返回格式
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader):
        # 数据送给模型
        y_out = my_model(input_ids, attention_mask, token_type_ids)
        print('y_out---->', y_out.shape, y_out)
        break


# todo:4-模型训练
def train():
    # 实例化数据源 通过训练文件
    my_dataset_train = load_dataset('csv', data_files='data/train.csv', split="train")

    # 实例化数据迭代器对象my_dataloader
    my_dataloader = DataLoader(dataset=my_dataset_train,
                               batch_size=8,
                               collate_fn=collate_fn,
                               shuffle=True,
                               drop_last=True)

    # 实例化下游任务模型my_model
    my_model = MyModel().to(device)

    # 实例化优化器my_optimizer
    # AdamW: Adam优化版, 引入权重衰减策略 L2正则化
    my_optimizer = AdamW(my_model.parameters(), lr=5e-4)

    # 实例化损失函数my_criterion
    my_criterion = nn.CrossEntropyLoss()

    # 不训练预训练模型 只让预训练模型计算数据特征 不需要计算梯度
    for param in my_model_pretrained.parameters():
        param.requires_grad_(False)

    # 设置训练参数
    epochs = 3

    # 设置模型为训练模型
    my_model.train()

    # 外层for循环 控制轮数
    for epoch_idx in range(epochs):

        # 每次轮次开始计算时间
        starttime = int(time.time())

        # 内层for循环 控制迭代次数
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader, start=1):

            # 给模型喂数据 [8,500] --> [8,2]
            my_out = my_model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

            # 计算损失
            my_loss = my_criterion(my_out, labels)

            # 梯度清零
            my_optimizer.zero_grad()

            # 反向传播
            my_loss.backward()

            # 参数更新
            my_optimizer.step()

            # 每5次迭代 算一下准确率
            if i % 5 == 0:
                out = my_out.argmax(dim=1)  # [8,2] --> (8,)
                acc = (out == labels).sum().item() / len(labels)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d' \
                      % (epoch_idx, i, my_loss.item(), acc, int(time.time()) - starttime))

        # 每个轮次保存模型
        # 字符串格式化输出 print('%d' % (1))
        torch.save(my_model.state_dict(), 'train_model/my_model_class_%d.bin' % (epoch_idx + 1))


# todo:5-模型推理
def inference():
    # 实例化数据源对象my_dataset_test
    print('\n加载测试集')
    my_dataset_test = load_dataset('csv', data_files='data/test.csv', split='train')
    print('my_dataset_test--->', my_dataset_test)
    # print(my_dataset_test[0:3])

    # 实例化化my_dataloader
    my_loader_test = DataLoader(my_dataset_test,
                                batch_size=8,
                                collate_fn=collate_fn,
                                shuffle=True,
                                drop_last=True)

    # 实例化下游任务模型my_model
    path = 'train_model/my_model_class_3.bin'
    my_model = MyModel().to(device)
    my_model.load_state_dict(torch.load(path))
    print('my_model-->', my_model)

    # 设置下游任务模型为评估模式
    my_model.eval()

    # 设置评估参数
    correct = 0
    total = 0

    # 给模型送数据 测试预测结果
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_loader_test):

        # 预训练模型进行特征抽取
        with torch.no_grad():
            my_out = my_model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        # 贪心算法求预测结果
        out = my_out.argmax(dim=1)

        # 计算准确率
        # 预测正确的样本数
        correct += (out == labels).sum().item()
        # 总样本数
        total += len(labels)

        # 每5次迭代打印一次准确率
        if i % 5 == 0:
            print(correct / total, end=" ")
            print(my_tokenizer.decode(input_ids[0], skip_special_tokens=True), end=" ")
            print('预测值 真实值:', out[0].item(), labels[0].item())


if __name__ == '__main__':
    # train()
    inference()
