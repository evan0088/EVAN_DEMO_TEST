import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score
from tqdm import tqdm
from config import Config
from utils import build_dataloader, get_time_diff
from bert_classifer_model import BertClassifier

import warnings
warnings.filterwarnings("ignore")


def model2train():
    # 初始化最佳验证 F1 分数，用于保存性能最好的模型
    best_dev_f1 = 0.0

    # 1. 加载训练、测试和验证数据集的 DataLoader
    train_loader, test_loader, dev_loader = build_dataloader()

    # 2. 初始化 BERT 分类模型
    model = BertClassifier().to(conf.device)

    # 3. 定义优化器（AdamW，适合 Transformer 模型）和损失函数（交叉熵）
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 4.1 遍历每个训练轮次（epoch）
    for epoch in range(conf.num_epochs):
        # 初始化参数
        total_loss = 0  # 累计训练损失
        train_preds, train_labels = [], []  # 存储训练集预测和真实标签

        # 4.2 遍历训练 DataLoader 进行模型训练
        for batch in tqdm(train_loader, desc=f"Bert Classifier Training Epoch {epoch + 1}/{conf.num_epochs}...."):
            print("len(batch)--->",len(batch))
            model.train()
            # 5.提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(conf.device), attention_mask.to(conf.device), labels.to(
                conf.device)

            # 5.1 前向传播
            logits = model(input_ids, attention_mask)
            # 5.2 梯度归零
            optimizer.zero_grad()
            # 5.3 损失计算
            loss = criterion(logits, labels)
            # 5.4 反向传播
            loss.backward()
            # 5.5 参数更新
            optimizer.step()

            # 6.1 累计损失
            total_loss += loss.item()

            # 6.2 获取预测结果（最大 logits 对应的类别）
            preds = torch.argmax(logits, dim=1)

            # 6.3 存储预测和真实标签，用于计算训练集指标
            train_preds.extend(preds.tolist())
            train_labels.extend(labels.tolist())

            # 7.每 10 个批次或非空批次时，打印训练信息并评估验证集
            if len(batch) % 10 == 0 or len(batch) != 0:
                print(f"Epoch {epoch + 1}/{conf.num_epochs}")
                print(f"Train Loss: {total_loss / len(train_loader):.4f}")
                # 8.1 在验证集上评估模型
                report, f1score, accuracy, precision = model2dev(model, dev_loader, conf.device)
                print(f"Dev F1: {f1score:.4f}")
                print(f"Dev Accuracy: {accuracy:.4f}")

                # 8.2 如果验证 F1 分数优于历史最佳，保存模型
                if f1score > best_dev_f1:
                    best_dev_f1 = f1score
                    torch.save(model.state_dict(), conf.model_save_path)
                    print("模型保存！！")

        # 8.3 计算并打印训练集的分类报告
        train_report = classification_report(train_labels, train_preds, target_names=conf.class_list, output_dict=True)
        print(train_report)


def model2dev(model, data_loader, device):
    # 1. 设置模型为评估模式（禁用 dropout 和 batch norm）
    model.eval()

    # 2. 初始化列表，存储预测结果和真实标签
    preds, true_labels = [], []

    # 3. 禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        # 4. 遍历数据加载器，逐批次进行预测
        for batch in tqdm(data_loader, desc="Bert Classifier Evaluating ......"):
            # 4.1 提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # 4.2 前向传播：模型预测
            logits = model(input_ids, attention_mask)

            # 4.3 获取预测结果（最大 logits 对应的类别）
            batch_preds = torch.argmax(logits, dim=1)

            # 4.4 存储预测和真实标签
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 5. 计算分类报告、F1 分数、准确度和精确度
    report = classification_report(true_labels, preds)
    f1score = f1_score(true_labels, preds, average='micro')  # 使用微平均计算 F1 分数
    accuracy = accuracy_score(true_labels, preds)  # 计算准确度
    precision = precision_score(true_labels, preds, average='micro')  # 使用微平均计算精确度

    # 6. 返回评估结果
    return report, f1score, accuracy, precision


if __name__ == '__main__':
    # 1.加载配置对象，包含模型参数、路径等
    conf = Config()
    # 2.模型训练
    model2train()

    # # 3.模型评估
    # # 3.1 加载测试集数据
    # train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # # 3.2 初始化 BERT 分类模型
    # model = BertClassifier()
    # # 3.3 加载预训练模型权重
    # model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    # # 3.4 在测试集上评估模型
    # test_report, f1score, accuracy, precision = model2dev(model, test_dataloader, conf.device)
    # # 3.5 打印测试集评估结果
    # print("Test Set Evaluation:")
    # print(f"Test F1: {f1score:.4f}")
    # print("Test Classification Report:")
    # print(test_report)
