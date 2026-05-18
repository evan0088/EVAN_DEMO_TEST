import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
# 特征工程 TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# 评估 Acc P R f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report
# 导入RF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from tqdm import tqdm  # 引入 tqdm 用于进度条

pd.set_option('display.expand_frame_repr', False)  # 避免宽表格换行
pd.set_option('display.max_columns', None)  # 确保所有列可见

# 1.配置文件读取
from config import Config

conf = Config()

# 第一步：读取数据
# 读取训练数据集
df = pd.read_csv(conf.processt_train_datapath)[:20000]  # 演示
print(df.head(5))
words = df["words"]
labels = df["label"]
print(df.head(5))

# 第二步：将文本转换为数值特征
# 读取停用词文件
stop_words = open(conf.stop_words_path, 'r', encoding='utf-8').read().split()
transfer = TfidfVectorizer(stop_words=stop_words)
words_features = transfer.fit_transform(words)

# # 1.查看特征长度
print(f'特征：{words_features}')
#
# # 2.特征维度 形状
print(f'特征维度：{words_features.shape}')
# #
# # 3.特征名字
print(f'特征名字：{transfer.get_feature_names_out()}')
# #
# # 4.特征词表
print(f'特征词表vocab:{transfer.vocabulary_}')


# 第三步 模型训练  word_features  labels
# 1.数据集切分 train_test_split
x_train, x_test, y_train, y_test = train_test_split(words_features, labels, test_size=0.2)

# 2.实例化模型  RF
model = RandomForestClassifier(n_estimators=100)

# 3.模型训练 RF  .fit
model.fit(x_train, y_train)

# 4.模型预测 .predict
y_pred = model.predict(x_test)

# 5.模型评估  .score Acc  P  R  F1 AUC Classification Report
acc_score = accuracy_score(y_test, y_pred)
# 报错提示  默认 average='binary' ，希望多分类
p_score = precision_score(y_test, y_pred, average='micro')
r_score = recall_score(y_test, y_pred, average='micro')
cls_report = classification_report(y_test, y_pred)
print(f'分类报告：\n{cls_report}')
print(f'Acc：\n{acc_score}')
print(f'精确度：\n{p_score}')
print(f'召回率：\n{r_score}')  # 医疗---漏检

pickle.dump(model, open(conf.rf_model_save_path+"/rf_model.pkl", 'wb'), 3)  # pickle 保存报错

# #  查看特征,features是稀疏矩阵，稀疏矩阵的格式是 (row, column) value
# print(features)
# #  查看特征维度  1000行 4687
# print(features.shape)
# print(list(tfidf.get_feature_names_out()))
# print(len(tfidf.get_feature_names_out()))
# print(tfidf.vocabulary_)
# print(len(tfidf.vocabulary_))

# # 第三步：划分训练集和测试集，模型训练和模型预测评估
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=22)
#
# # 训练随机森林模型
# model = RandomForestClassifier()  # 设置 verbose=1 以输出训练进度
# print("训练模型...")
# # 使用 tqdm 包装 model.fit 来显示进度条
# for _ in tqdm(range(1), desc="RandomForest模型训练进度...."):
#     model.fit(x_train, y_train)
#
# # 模型预测并评估
# print("模型预测评估...")
# y_pred = model.predict(x_test)
# print("预测结果:", y_pred)
# print("准确率:", accuracy_score(y_test, y_pred))
# print("精确率 (micro):", precision_score(y_test, y_pred, average='micro'))
# print("召回率 (micro):", recall_score(y_test, y_pred, average='micro'))
# print("F1分数 (micro):", f1_score(y_test, y_pred, average='micro'))
#
# # 第四步：保存模型和向量化器
# print("保存模型和向量化器...")
# with open(conf.rf_model_save_path + '/rf_model_.pkl', 'wb') as f:
#     pickle.dump(model, f)
# with open(conf.rf_model_save_path + '/tfidf_vectorizer_.pkl', 'wb') as f:
#     pickle.dump(tfidf, f)
#
# print("模型和向量化器，保存成功！")
