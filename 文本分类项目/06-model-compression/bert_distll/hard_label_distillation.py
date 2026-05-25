import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score
from tqdm import tqdm
from utils import build_dataloader, get_time_diff
from bert_classifer_model import BertClassifier
from bilstm_classifier import BiLSTMClassifier
from config import Config
import time

import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息


def model2train(teacher_model, student_model, train_loader, dev_loader):
    """
    训练学生模型（BiLSTM）使用硬标签蒸馏，学习教师模型（BERT）的预测类别。

    参数：
        teacher_model: 教师模型（BERT），提供硬标签。
        student_model: 学生模型（BiLSTM），需要学习教师模型的预测。
        train_loader: 训练数据加载器，提供训练数据批次。
        dev_loader: 验证数据加载器，提供验证数据批次。
    """
    # 初始化参数
    best_dev_f1 = 0.0  # 记录最佳验证 F1 分数
    step = 0  # 训练步数计数器
    patience = 3  # 早停耐心值
    epochs_no_improve = 0  # 记录未提升的 epoch 数

    # 1.初始化优化器和损失函数
    optimizer = AdamW(student_model.parameters(), lr=conf.learning_rate)  # 使用 AdamW 优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，用于硬标签损失

    # 2.1 遍历每个epoch
    for epoch in range(conf.num_epochs):
        student_model.train()  # 设置学生模型为训练模式
        teacher_model.eval()  # 设置教师模型为评估模式（不更新权重）
        total_loss = 0  # 记录当前 epoch 的总损失
        train_preds, train_labels = [], []  # 记录训练预测和真实标签
        epoch_start_time = time.time()  # 记录 epoch 开始时间

        print(f"\n硬标签蒸馏训练 Hard Label Distillation Epoch {epoch + 1}/{conf.num_epochs}...")
        # 2.2 遍历训练数据批次
        for batch in tqdm(train_loader, desc=f"Hard Label Distillation Epoch {epoch + 1}/{conf.num_epochs}"):
            step_start_time = time.time()  # 记录当前 step 开始时间
            input_ids, attention_mask, labels = batch  # 获取输入数据
            input_ids, attention_mask, labels = input_ids.to(conf.device), attention_mask.to(conf.device), labels.to(
                conf.device)

            # 3.1.1 获取教师模型的预测（硬标签）
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)
                teacher_preds = torch.argmax(teacher_logits, dim=1)
            # 3.1.2 获取学生模型的输出 logits
            student_logits = student_model(input_ids, attention_mask)

            # 3.2 计算硬标签损失（交叉熵，使用教师模型的预测）
            loss = criterion(student_logits, teacher_preds)

            # 3.3 梯度归零
            optimizer.zero_grad()
            # 3.4 反向传播
            loss.backward()
            # 3.5 参数更新
            optimizer.step()

            total_loss += loss.item()  # 累加损失

            # 4.记录预测结果
            preds = torch.argmax(student_logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            step += 1  # 步数加 1
            step_duration = time.time() - step_start_time  # 计算 step 耗时

            # 5.每 10 个 step 验证一次
            if step % 10 == 0:
                student_model.eval()  # 切换到评估模式
                avg_loss = total_loss / (len(train_preds) / train_loader.batch_size)  # 计算平均损失
                report, f1score, accuracy, precision = model2dev(student_model, dev_loader, conf.device)  # 验证
                print(f"Step {step}, Epoch {epoch + 1}/{conf.num_epochs}")
                print(f"Step Duration: {step_duration:.2f}s")
                print(f"Train Loss: {avg_loss:.4f}")
                print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
                print(f"Dev Precision: {precision:.4f}")
                print(f"Dev 分类报告:\n{report}")
                student_model.train()  # 切换回训练模式

        # 6.1 计算训练集指标
        train_report = classification_report(train_labels, train_preds)

        # 6.2 验证（每个 epoch 结束时）
        student_model.eval()
        report, f1score, accuracy, precision = model2dev(student_model, dev_loader, conf.device)

        # 7.计算 epoch 耗时
        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1}/{conf.num_epochs}")
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Train 分类报告: {train_report}")
        print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
        print(f"Dev Precision: {precision:.4f}")
        print(f"Dev 分类报告:\n{report}")

        # 8.保存最佳模型并检查早停
        if f1score > best_dev_f1:
            best_dev_f1 = f1score
            torch.save(student_model.state_dict(), conf.distil_h_model_save_path)
            print("模型保存！！")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Dev F1 未提升，当前未提升 epoch 数: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"早停触发！Dev F1 在 {patience} 个 epoch 内未提升，停止训练。")
                break

        student_model.train()


def model2dev(model, data_loader, device):
    model.eval()
    preds, true_labels = [], []

    # 1.关闭梯度计算
    with torch.no_grad():
        # 2.遍历数据
        for batch in tqdm(data_loader, desc="BiLSTM Classifier Evaluating ......"):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # 3.前向传播
            logits = model(input_ids, attention_mask)
            # 4.获取模型输出 logits
            batch_preds = torch.argmax(logits, dim=1)

            # 收集预测和真实标签
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        # 计算分类报告和指标
        report = classification_report(true_labels, preds)
        f1score = f1_score(true_labels, preds, average='micro')
        accuracy = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds, average='micro')

    return report, f1score, accuracy, precision


if __name__ == '__main__':
    # 记录训练开始时间
    start_time = time.time()

    # 1.加载配置文件
    conf = Config()  # 加载配置文件

    # 2.加载教师训练数据与学生训练数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 3.定义教师模型,加载模型参数
    teacher_model = BertClassifier().to(conf.device)
    teacher_model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))

    # 4.定义学生模型
    student_model = BiLSTMClassifier(conf).to(conf.device)

    # 5.模型训练
    model2train(teacher_model, student_model, train_dataloader, dev_dataloader)

    # 6.获取训练耗时
    total_training_duration = get_time_diff(start_time)
    print(f"硬标签蒸馏训练耗时： {total_training_duration}")
