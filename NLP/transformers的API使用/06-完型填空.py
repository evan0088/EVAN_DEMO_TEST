import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import time

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载字典和分词工具
my_tokenizer = BertTokenizer.from_pretrained('model/bert-base-chinese')

# 加载预训练模型
my_model_pretrained = BertModel.from_pretrained('model/bert-base-chinese').to(device)

# 查看预训练模型的输出维度
hidden_size = my_model_pretrained.config.hidden_size
print('hidden_size--->', hidden_size)  # 768


# todo:1-加载数据集
def dm_file2dataset():
    # 获取训练数据集
    train_dataset_tmp = load_dataset('csv', data_files='data/train.csv', split='train')
    print('train_dataset_tmp--->', train_dataset_tmp)
    print('train_dataset_tmp[0]--->', train_dataset_tmp[0])

    # 过滤掉样本的评论内容长度小于等于32的样本
    # x->{'label':0, 'text':xxxx}
    my_train_dataset = train_dataset_tmp.filter(lambda x: len(x['text']) > 32)
    print('my_train_dataset--->', my_train_dataset)

    # 获取测试数据集
    test_dataset_tmp = load_dataset('csv', data_files='data/test.csv', split='train')
    my_test_dataset = test_dataset_tmp.filter(lambda x: len(x['text']) > 32)

    return my_train_dataset, my_test_dataset


# todo:2-构建数据加载器  自定义函数
# 数据集处理自定义函数
def collate_fn(data):
    # data -> [{'label':0, 'text':xxxx}, {'label':1, 'text':xxxx}, ...]
    # print('data--->', data)
    # 获取批次样本的句子列表
    sents = [i['text'] for i in data]
    # print('sents--->', sents)

    # 文本数值化
    data = my_tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=32,
                                          return_tensors='pt')
    # print('data--->', data)

    # input_ids 编码之后的数字
    # attention_mask 是补零的位置是0,其他位置是1
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)

    # 取出每批的8个句子 在第17个位置clone出来 做真实标签
    labels = input_ids[:, 16].clone()
    # tmpa = input_ids[:, 16]
    # print('tmpa--->', tmpa, tmpa.shape)       # torch.Size([8])
    # print('labels-->', labels.shape, labels)  # torch.Size([8])

    # 将第17个词替换成[MASK]的下标值
    # 获取[MASK]字符
    # print(my_tokenizer.mask_token)
    # 获取[MASK]字符的下标
    # print(my_tokenizer.mask_token_id)
    # print(my_tokenizer.get_vocab()[my_tokenizer.mask_token])
    # input_ids[:, 16] = my_tokenizer.get_vocab()[my_tokenizer.mask_token]
    input_ids[:, 16] = my_tokenizer.mask_token_id
    # print('input_ids--->', input_ids)
    return input_ids, attention_mask, token_type_ids, labels


# 数据源 数据迭代器 测试
def dm01_test_dataset():
    # 生成数据源dataset对象
    dataset_train_tmp = load_dataset('csv', data_files='data/train.csv', split="train")
    # print('dataset_train_tmp--->', dataset_train_tmp)

    # 按照条件过滤数据源对象
    my_dataset_train = dataset_train_tmp.filter(lambda x: len(x['text']) > 32)
    # print('my_dataset_train--->', my_dataset_train)
    # print('my_dataset_train[0:3]-->', my_dataset_train[0:3])

    # 通过dataloader进行迭代
    my_dataloader = DataLoader(my_dataset_train,
                               batch_size=8,
                               collate_fn=collate_fn,
                               shuffle=True,
                               drop_last=True)
    # print('my_dataloader--->', my_dataloader)

    # 调整数据迭代器对象数据返回格式
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader):
        print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

        print('\n第1句mask的信息')
        print(my_tokenizer.decode(input_ids[0]))
        print(my_tokenizer.decode(labels[0]))

        print('\n第2句mask的信息')
        print(my_tokenizer.decode(input_ids[1]))
        print(my_tokenizer.decode(labels[1]))
        break


# todo:3-构建下游任务的网络模型
# 定义下游任务模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层
        self.fc = nn.Linear(768, my_tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 预训练模型不进行训练
        with torch.no_grad():
            out = my_model_pretrained(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)

        # 第17个token的语义表示经过全连接层
        # 下游任务进行训练 形状[8,768] ---> [8, 21128]
        # print('out.last_hidden_state[:, 16, :]--->', out.last_hidden_state[:, 16, :])
        # print('out.last_hidden_state[:, 16]--->', out.last_hidden_state[:, 16])
        out = self.fc(out.last_hidden_state[:, 16])
        # 返回
        return out


# 模型输入和输出测试
def dm02_test_mymodel():
    # 生成数据源dataset对象
    dataset_train_tmp = load_dataset('csv', data_files='data/train.csv', split="train")
    # print('dataset_train_tmp--->', dataset_train_tmp)

    # 按照条件过滤数据源对象
    my_dataset_train = dataset_train_tmp.filter(lambda x: len(x['text']) > 32)
    # print('my_dataset_train--->', my_dataset_train)
    # print('my_dataset_train[0:3]-->', my_dataset_train[0:3])

    # 通过dataloader进行迭代
    my_dataloader = DataLoader(my_dataset_train,
                               batch_size=8,
                               collate_fn=collate_fn,
                               shuffle=True,
                               drop_last=True)
    print('my_dataloader--->', my_dataloader)

    # 不训练,不需要计算梯度
    for param in my_model_pretrained.parameters():
        param.requires_grad_(False)

    # 实例化下游任务模型
    mymodel = MyModel().to(device)

    # 调整数据迭代器对象数据返回格式
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader):
        # print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

        print('\n第1句mask的信息')
        print(my_tokenizer.decode(input_ids[0]))
        print(my_tokenizer.decode(labels[0]))

        print('\n第2句mask的信息')
        print(my_tokenizer.decode(input_ids[1]))
        print(my_tokenizer.decode(labels[1]))

        # 给模型喂数据 [8,768] ---> [8,21128] 填空就是分类 21128个单词中找一个单词
        myout = mymodel(input_ids, attention_mask, token_type_ids)
        print('myout--->', myout.shape, myout)
        break


# todo:4-模型训练
# 模型训练 - 填空
def dm03_train_model():
    # 实例化数据源对象my_dataset_train
    dataset_train_tmp = load_dataset('csv', data_files='data/train.csv', split="train")
    my_dataset_train = dataset_train_tmp.filter(lambda x: len(x['text']) > 32)
    print('my_dataset_train--->', my_dataset_train)

    # 实例化数据迭代器对象my_dataloader
    my_dataloader = DataLoader(my_dataset_train,
                               batch_size=8,
                               collate_fn=collate_fn,
                               shuffle=True,
                               drop_last=True)

    # 实例化下游任务模型my_model
    my_model = MyModel().to(device)

    # 实例化优化器my_optimizer
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
        starttime = int(time.time())
        # 内层for循环 控制迭代次数
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader, start=1):
            # 给模型喂数据 [8,32] --> [8,21128]
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
            if i % 20 == 0:
                out = my_out.argmax(dim=1)  # [8,21128] --> (8,)
                acc = (out == labels).sum().item() / len(labels)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d' \
                      % (epoch_idx, i, my_loss.item(), acc, int(time.time()) - starttime))

        # 每个轮次保存模型
        torch.save(my_model.state_dict(), 'train_model/my_model_mask_%d.bin' % (epoch_idx + 1))


# todo:5-模型推理
# 模型测试：填空
def dm04_evaluate_model():
    # 实例化数据源对象my_dataset_test
    print('\n加载测试集')
    my_dataset_tmp = load_dataset('csv', data_files='data/test.csv', split='train')
    my_dataset_test = my_dataset_tmp.filter(lambda x: len(x['text']) > 32)
    print('my_dataset_test--->', my_dataset_test)
    # print(my_dataset_test[0:3])

    # 实例化化dataloader
    my_loader_test = DataLoader(my_dataset_test,
                                batch_size=8,
                                collate_fn=collate_fn,
                                shuffle=True,
                                drop_last=True)

    # 实例化下游任务模型my_model
    path = 'train_model/my_model_mask_3.bin'
    my_model = MyModel().to(device)
    my_model.load_state_dict(torch.load(path))
    print('my_model-->', my_model)

    # 设置下游任务模型为评估模式
    my_model.eval()

    # 设置评估参数
    correct = 0
    total = 0

    # 给模型送数据 测试预测结果
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(my_loader_test):
        with torch.no_grad():
            my_out = my_model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = my_out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

        if i % 5 == 0:
            print(i + 1, my_tokenizer.decode(input_ids[0]))
            print('预测值:', my_tokenizer.decode(out[0]), '\t真实值:', my_tokenizer.decode(labels[0]))
            print(correct / total)


if __name__ == '__main__':
    # dm01_test_dataset()
    # dm02_test_mymodel()
    # dm03_train_model()
    dm04_evaluate_model()