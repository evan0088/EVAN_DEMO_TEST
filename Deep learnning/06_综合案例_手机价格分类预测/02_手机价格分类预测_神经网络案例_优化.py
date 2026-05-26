# TODO 导包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import time
# 优化: 导入标准化包,提前对数据做标准化处理
from sklearn.preprocessing import StandardScaler


# TODO 1.自定义函数,做数据加载和处理
def get_data_loader(batch_size):
    # 1.加载数据
    data = pd.read_csv('data/手机价格预测.csv')
    # 2.了解数据
    print(data.shape)
    # print(data.columns)
    # print(data.head())
    # data.info()
    # 3.处理数据
    # 3.1获取特征值x和目标值y
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # 3.2为了后续用于张量,提前做类型转换：特征值转浮点，目标值转整型
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # 3.3 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88)
    # TODO StandardScaler添加数据标准化处理
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    # 3.4 构建数据集,最终为训练集dataloader和测试集dataloader
    # 先把numpy数据集转换成张量,然后封装成张量数据集
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid.values))
    # 再把张量数据集封装成数据加载器,并且设置批量大小和是否打乱数据
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # 4.返回结果: 数据加载器,数据集长度,输入特征维度,输出特征维度
    return train_dataloader, valid_dataloader, x_train.shape[1], len(y.unique())


# TODO 2.自定义类,构建神经网络模型
class PhonePriceModel(torch.nn.Module):
    # 重写__init__方法和forward方法
    def __init__(self, input_num, output_num):
        # 调用父类的构造方法
        super().__init__()
        # 定义网络结构
        self.linear1 = torch.nn.Linear(input_num, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        # TODO 优化3: 增加网络深度
        self.linear3 = torch.nn.Linear(256, 256)
        self.out = torch.nn.Linear(256, output_num)

    # 重写forward方法
    def forward(self, x):
        # 加权求和->激活函数(默认隐藏层都用relu作为激活函数)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # TODO 优化3: 增加网络深度,后添加计算
        x = torch.relu(self.linear3(x))
        # 输出层(注意:后续使用交叉熵损失函数,它已经自带了softmax()操作,此处只需要做加权求和操作)
        x = self.out(x)
        # 返回最后的加权求和结果,不是预测值
        return x


# TODO 3.模型训练
def train_model(train_dataloader, model, epochs):
    # 1.获取数据(此处参数已经传入)
    # 2.获取模型(此处参数已经传入)
    # 3.创建损失函数对象
    loss_fn = torch.nn.CrossEntropyLoss()
    # 4.创建优化器对象
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # TODO 优化5: SGD-> Adam, 同时调整学习率
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.0001)
    # 5.循环训练模型
    # TODO 如果后续使用了随机失活dropout,此处模型就需要切换到训练模式
    model.train()
    # 外层循环轮次,内层循环批次
    for epoch in range(epochs):
        # 定义初始参数
        total_loss, batch_cnt, start = 0.0, 0, time.time()
        for batch_x, batch_y in train_dataloader:
            # 正(前)向传播:从输入到输出: 预测值和损失值
            # 模型预测(此处拿到了加权求和结果)
            y = model(batch_x)  # 底层自动调用了forward
            # 损失计算 (此处先底层调用了softmax获取预测值,然后计算损失值)
            loss = loss_fn(y, batch_y)
            # 累加损失值和批次数
            total_loss += loss.item()
            batch_cnt += 1
            # 反向传播:从输出到输入: 梯度计算和参数更新
            # 梯度清零!!!
            optimizer.zero_grad()
            # 梯度计算
            loss.backward()
            # 参数更新
            optimizer.step()
        # 走到此处,说明一轮结束: 累加损失和批次数用于计算每轮损失值.
        epoch_loss = total_loss / batch_cnt
        print(f"第{epoch + 1}轮,运行时间{time.time() - start:.2f}秒,损失值为:{epoch_loss:.2f}")
    # 6.保存训练好的模型参数字典
    torch.save(model.state_dict(), 'model/手机价格分类预测_优化.pth')


# TODO 4.模型评估
def eval_model(valid_dataloader, input_num, output_num):
    # 1.获取数据(此处参数已经传入)
    # 2.创建新模型对象,加载训练好的参数字典,用于评估
    model = PhonePriceModel(input_num, output_num)
    model.load_state_dict(torch.load('model/手机价格分类预测_优化.pth'))
    # 3.定义变量,记录: 预测正确的样本数.
    correct = 0
    # TODO 如果后续使用了随机失活dropout,此处模型就需要切换到测试模式
    model.eval()
    # 4.具体的 每批评估 过程.
    for batch_x, batch_y in valid_dataloader:
        # 4.1 模型预测(此处拿到的是加权求和结果)
        y = model(batch_x)
        # 4.2 获取预测结果
        y_pred = torch.argmax(y, dim=1)
        # 4.3 获取预测正确的个数
        correct += (y_pred == batch_y).sum()

    # 5.求预测精度
    print(f'Acc: {(correct / len(valid_dataloader.dataset)):.4f}')


# 程序的主入口
if __name__ == '__main__':
    # 1.自定义函数,做数据加载和处理
    # 设置batch_size大小并传参给get_data_loader()函数
    # TODO 优化1: 调整batch_size大小
    batch_size = 16
    # TODO 优化2: tandardScaler提前对数据做标准化处理
    train_dataloader, valid_dataloader, input_num, output_num = get_data_loader(batch_size)
    # 2.自定义类,构建神经网络模型
    # TODO 优化3: 增加网络深度
    model = PhonePriceModel(input_num, output_num)
    # 3.模型训练(正向/反向传播)
    # TODO 优化4: 调整训练轮次
    epochs = 100
    # TODO 优化5: SGD-> Adam,同时调整学习率
    train_model(train_dataloader, model, epochs)
    # 4.模型评估
    eval_model(valid_dataloader, input_num, output_num)
