# TODO 导包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import time


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
    # 3.1获取特征值x-20列和目标值y-1列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # 3.2为了后续用于张量,提前做类型转换：特征值转浮点，目标值转整型
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # 3.3 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88)
    # 3.4 todo 构建数据集,最终为训练集dataloader和测试集dataloader
    # 先把numpy数据集转换成张量,然后封装成张量数据集
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid.values), torch.from_numpy(y_valid.values))
    # 再把张量数据集封装成数据加载器,并且设置批量大小和是否打乱数据
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # 4.返回结果: 数据加载器,数据集长度,输入特征维度,输出特征维度0,1,2,3
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
        self.out = torch.nn.Linear(256, output_num)

    # 重写forward方法
    def forward(self, x):
        # 加权求和->激活函数(默认隐藏层都用relu作为激活函数)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # 输出层(todo 注意:后续使用交叉熵损失函数,它已经自带了softmax()操作,此处只需要做加权求和操作)
        x = self.out(x)
        # TODO 返回最后的加权求和结果,不是预测值
        return x


# TODO 3.模型训练
def train_model(train_dataloader, model, epochs):
    # 1.获取数据(此处参数已经传入)
    # 2.获取模型(此处参数已经传入)
    # 3.创建多分类损失函数对象
    loss_fn = torch.nn.CrossEntropyLoss()
    # 4.创建优化器对象
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # 5.循环训练模型
    # TODO 外层循环轮次,内层循环批次
    for epoch in range(epochs):
        # 定义初始参数
        total_loss, batch_cnt, start = 0.0, 0, time.time()
        for batch_x, batch_y in train_dataloader:
            # TODO 正(前)向传播:从输入到输出: 预测值和损失值
            # 模型预测(此处拿到了加权求和结果)
            y = model(batch_x)  # todo 底层自动调用了forward
            # 损失计算 (此处先底层调用了softmax获取预测值,然后计算损失值)
            loss = loss_fn(y, batch_y)
            # 累加损失值和批次数
            total_loss += loss.item()
            batch_cnt += 1
            # TODO 反向传播:从输出到输入: 梯度计算和参数更新
            # 梯度清零!!!
            optimizer.zero_grad()
            # 梯度计算
            loss.backward()  # 自动微分
            # 参数更新
            optimizer.step()
        # TODO 走到此处,说明一轮结束: 累加损失和批次数用于计算每轮损失值.
        epoch_loss = total_loss / batch_cnt
        print(f"第{epoch + 1}轮,运行时间{time.time() - start:.2f}秒,损失值为:{epoch_loss:.2f}")
    # 6.保存训练好的模型参数字典
    torch.save(model.state_dict(), 'model/手机价格分类预测_原生.pth')


# TODO 4.模型评估
def eval_model(valid_dataloader, input_num, output_num):
    # 1.获取数据(此处参数已经传入)
    # 2.创建新模型对象,加载训练好的参数字典,用于评估
    model = PhonePriceModel(input_num, output_num)
    model.load_state_dict(torch.load('model/手机价格分类预测_原生.pth'))
    # 3.定义变量,记录: 预测正确的样本数.
    correct = 0
    # 4.具体的 每批评估 过程.
    for batch_x, batch_y in valid_dataloader:
        # 4.1 模型预测(此处拿到的是加权求和结果)
        y = model(batch_x)
        # 4.2 argmax()最大的获取预测结果
        y_pred = torch.argmax(y, dim=1)
        # 4.3 获取预测正确的个数
        correct += (y_pred == batch_y).sum()

    # 5.求预测精度
    print(f'Acc: {(correct / len(valid_dataloader.dataset)):.4f}')


# 程序的主入口
if __name__ == '__main__':
    # TODO 1.自定义函数,做数据加载和处理
    # 设置batch_size大小并传参给get_data_loader()函数
    batch_size = 8
    train_dataloader, valid_dataloader, input_num, output_num = get_data_loader(batch_size)
    # TODO 2.自定义类,构建神经网络模型
    model = PhonePriceModel(input_num, output_num)
    # TODO 3.模型训练(正向/反向传播)
    epochs = 50
    # train_model(train_dataloader, model, epochs)
    # TODO 4.模型评估
    eval_model(valid_dataloader, input_num, output_num)
