import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from config import Config
from utils import build_dataloader, get_time_diff
from bert_classifer_model import BertClassifier
from bilstm_classifier import BiLSTMClassifier
import time
from hard_label_distillation import model2dev

import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息
conf = Config()

def model2train():
    # 配置参数信息
    T = 2.0  # 温度参数，用于软标签蒸馏
    alpha = 0.7  # 软标签和硬标签损失的权重
    step = 0  # 训练步数计数器
    best_dev_f1 = 0.0  # 记录最佳验证 F1 分数

    # 1.教师训练数据与学生训练数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 2.定义教师模型,加载模型参数
    teacher_model = BertClassifier().to(conf.device)
    teacher_model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))

    # 3.定义学生模型
    student_model = BiLSTMClassifier(conf).to(conf.device)

    # 4.初始化优化器和损失函数
    optimizer = AdamW(student_model.parameters(), lr=conf.learning_rate)  # 使用 AdamW 优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，用于硬标签损失

    # 5.1 遍历每个 epoch
    for epoch in range(conf.num_epochs):
        ## 设置学生模型为训练模式，设置教师模型为评估模式（不更新权重）
        student_model.train()
        teacher_model.eval()
        # 5.2 遍历训练数据批次
        for batch_index, (input_ids, attention_mask, labels) in enumerate(
                tqdm(train_dataloader, desc=f"软标签蒸馏训练的 Epoch {epoch + 1}/{conf.num_epochs}")):
            # 6. input_ids, attention_mask, label放到设备
            input_ids, attention_mask, labels = input_ids.to(conf.device), attention_mask.to(conf.device), labels.to(
                conf.device)
            # 6.1.1 获取教师模型的输出 logits软标签与教师模型的硬标签
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)
                teacher_preds = torch.argmax(teacher_logits, dim=1)  # 获取教师模型的硬标签
            # 6.1.2 获取学生模型的输出 logits
            student_logits = student_model(input_ids, attention_mask)

            # 6.2.1 计算软标签损失（KL 散度）
            teacher_log_probs = F.softmax(teacher_logits / T, dim=1)  # 教师模型的 概率
            student_log_probs = F.log_softmax(student_logits / T,
                                              dim=1)  # 学生模型的 log-概率 # 论文  softmax-T  预测值 真实值 1/(T*T)
            soft_loss = F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction='batchmean') * (T * T)
            # 6.2.2 计算硬标签损失（交叉熵，使用教师模型的预测）
            hard_loss = criterion(student_logits, teacher_preds)
            # 6.2.3 总损失：软标签和硬标签损失的加权和
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            # 6.3 梯度归零
            optimizer.zero_grad()
            # 6.4 反向传播
            loss.backward()  # 反向传播计算梯度
            # 6.5 参数更新
            optimizer.step()

            # 7. 每 2 个 batch 验证一次,batch级别验证model2dev
            if batch_index % 2 == 0:
                report, f1score, accuracy, precision = model2dev(student_model, dev_dataloader, conf.device)  # 验证
                print(f"Step {step}, Epoch {epoch + 1}/{conf.num_epochs} ===============批级别=============")
                print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
                print(f"Dev Precision: {precision:.4f}")
                print(f"Dev 分类报告:\n{report}")
                student_model.train()  # 切换回训练模式

                if f1score > best_dev_f1:
                    torch.save(student_model.state_dict(), conf.distil_s_model_save_path)

        # 8. epoch级别验证 model2dev
        report, f1score, accuracy, precision = model2dev(student_model, dev_dataloader, conf.device)
        print(f"\nEpoch {epoch + 1}/{conf.num_epochs}==============================epoch级别===========")
        print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
        print(f"Dev Precision: {precision:.4f}")
        print(f"Dev 分类报告:\n{report}")
        student_model.train()  # 切换回训练模式


if __name__ == "__main__":
    # 主程序：加载数据和模型，开始训练
    start_time = time.time()  # 记录训练开始时间
    # 1.加载配置文件
    conf = Config()
    # 2.模型训练
    model2train()
    # 3.训练总耗时
    total_training_duration = get_time_diff(start_time)  # 计算总训练耗时
    print(f"软标签蒸馏训练耗时： {total_training_duration}")
