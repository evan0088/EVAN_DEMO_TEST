# AI学习项目

本项目涵盖了机器学习、深度学习和自然语言处理(NLP)的核心知识和实践代码。

## 📚 全文知识网络（一图总览）

> 本仓库知识量大、章节多。下面这张"知识网络图"帮你快速建立全局认知：先看树状学科版图 → 再看概念依赖链 → 最后用章节地图反向定位。

### 一、三大学科版图（顶层视角）

```
                          AI 人工智能
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
机器学习 (ML)             深度学习 (DL)              自然语言处理 (NLP)
    │                         │                         │
    ├─ 监督学习                ├─ PyTorch 张量操作         ├─ 文本预处理（分词/清洗）
    │   ├─ 分类 (KNN)         ├─ Autograd 自动微分        ├─ 词表示 (One-hot/词向量)
    │   └─ 回归 (线性回归)     ├─ 激活函数                 ├─ RNN 家族
    │                         │   (Sigmoid/ReLU/Tanh)    │   (RNN/LSTM/GRU)
    ├─ 无监督学习              ├─ 5 步训练模板             │
    │   ├─ K-Means            └─ 神经网络结构             ├─ 注意力机制
    │   ├─ PCA                    (FFN/CNN/RNN)          │   (软/硬/加性/缩放点积)
    │   └─ DBSCAN                                        │
    │                                                    ├─ Transformer
    └─ 评估指标                                          │   (Encoder + Decoder)
        ├─ 回归: MAE/MSE/R²                              │
        ├─ 分类: 混淆矩阵/P-R-F1/AUC                     └─ 预训练模型
        └─ 损失: 交叉熵                                       (BERT / FastText / LLM)
```

### 一（补充）：ML / DL / NLP 的区别与升级关系

> 上面的树状图只回答了"有哪些"，没回答"为什么从 A 演进到 B"。本小节用 4 个角度把三者关系彻底讲清。

#### ① 三者的"包含关系"——一句话定位

```
        ┌──────────────────────────────────────┐
        │           AI（人工智能 = 大目标）       │
        │  ┌────────────────────────────────┐  │
        │  │   ML（机器学习 = 方法论的总称）  │  │
        │  │   ┌──────────────────────────┐ │  │
        │  │   │  DL（深度学习 = 神经网络派）│ │  │
        │  │   └──────────────────────────┘ │  │
        │  └────────────────────────────────┘  │
        └──────────────────────────────────────┘
                          ▲
                          │ 都可以拿来做的"任务领域"：
                          │
              ┌───────────┼───────────┐
              │           │           │
          NLP（语言）   CV（视觉）   语音/推荐 …
```

- **AI ⊃ ML ⊃ DL**：DL 是 ML 的一个子集——专门用神经网络的那一支
- **NLP 是任务领域**，不是方法论——它**可以用 ML 做（02-rf），也可以用 DL 做（03-fasttext / 04-bert）**
- 所以 **NLP 的发展史 = 先用 ML 做、再用 DL 做、最后用大模型做** 的升级史

#### ② 区别对照表——核心区别在"特征怎么来"

| 维度 | 🔧 传统机器学习 (ML) | 🤖 深度学习 (DL) |
|------|---------------------|------------------|
| **特征工程** | **人工设计**特征<br>(分词 / TF-IDF / N-gram / 关键词词典 / 大小写/链接数) | **模型自动学**特征<br>(Embedding / 卷积核 / 注意力权重) |
| **数据需求** | 几千~几万条就能跑出可用模型 | 几十万~亿级才能发挥优势 |
| **算力需求** | CPU 即可 | 必须 GPU/TPU |
| **可解释性** | 高（每个特征是人写出来的） | 低（特征是黑盒高维向量） |
| **典型模型** | KNN / SVM / 决策树 / 随机森林 / TF-IDF | CNN / RNN / LSTM / Transformer / BERT |
| **本仓库示例** | `02-rf` 随机森林 + TF-IDF | `03-fasttext` / `04-bert` / Transformer 全套 |

> 💡 **一句话区别**：ML 是"**人教模型怎么看**"——你要先告诉它"链接数 > 3 就可疑"；DL 是"**模型自己学怎么看**"——给它原文，它自己悟出"链接数"是个有用特征。

#### ③ NLP 的演进史——5 个时代的升级链

每一代都是为了**解决上一代的痛点**而生：

```
┌─────────────────────────────────────────────────────────────┐
│  时代1: 传统 ML 时代（对应 02-rf）                            │
│    流程: 分词 → TF-IDF/词袋 → 随机森林/SVM → 预测              │
│    痛点: 1) 必须手工分词、停用词、特征筛选                     │
│         2) 词与词之间没语义关系（"苹果"和"水果"完全无关）      │
│         3) 长文本被压成向量，丢失先后顺序                     │
└─────────────────────────────────────────────────────────────┘
                            ↓ 升级动力：让模型自己学语义
┌─────────────────────────────────────────────────────────────┐
│  时代2: 浅层神经网络（对应 03-fasttext）                      │
│    流程: 分词 → Embedding 词向量 → 平均池化 → 分类             │
│    解决了: 词向量带语义（"苹果"和"水果"距离很近）              │
│    痛点: 1) 还是看不到词的"先后顺序"（平均池化把顺序抹掉）     │
│         2) 模型很浅，复杂语义抓不住                           │
└─────────────────────────────────────────────────────────────┘
                            ↓ 升级动力：要看顺序、要更深
┌─────────────────────────────────────────────────────────────┐
│  时代3: 循环神经网络 RNN / LSTM / GRU                         │
│    流程: 词向量 → LSTM 逐字处理 → 末态隐藏向量 → 分类           │
│    解决了: 顺序信息被"记住"，能处理变长序列                    │
│    痛点: 1) 无法并行（必须一个字一个字按顺序算）               │
│         2) 长距离依赖会"遗忘"（句首信息传到句尾就糊了）        │
└─────────────────────────────────────────────────────────────┘
                            ↓ 升级动力：要并行、要看任意距离
┌─────────────────────────────────────────────────────────────┐
│  时代4: Transformer / BERT（对应 04-bert）                    │
│    流程: 词向量 + 位置编码 → 多头自注意力 → 任务头              │
│    解决了: 1) 完全并行计算（GPU 利用率拉满）                   │
│           2) 任意两个词都能"直接对话"（自注意力）              │
│           3) 预训练 + 微调 = 用很少数据就能做新任务            │
│    痛点: 1) 预训练成本极高（百万美元级）                       │
│         2) 每个新任务还是要微调                               │
└─────────────────────────────────────────────────────────────┘
                            ↓ 升级动力：连微调都不想做
┌─────────────────────────────────────────────────────────────┐
│  时代5: 大语言模型 LLM（对应 05-LLM）                         │
│    流程: 写好 Prompt → 调 GPT/DeepSeek API → 拿回答            │
│    解决了: 0 训练样本，靠提示词就能做几乎所有 NLP 任务         │
│    痛点: 1) 成本按 token 计费                                 │
│         2) 数据要走第三方 API                                 │
└─────────────────────────────────────────────────────────────┘
```

#### ④ 同一个任务，三种做法的真实对照（以"垃圾邮件分类"为例）

| 做法 | 训练数据量 | 特征怎么来 | 模型 | 优劣 | 本仓库参考 |
|------|----------|-----------|------|------|-----------|
| 🔧 **ML 派** | 几千封邮件 | **人工**：黑名单词、链接数、大写率、TF-IDF | 随机森林 | ✅ 可解释、CPU 跑得动<br>❌ 漏过新型套话 | `02-rf` 风格 |
| 🤖 **DL 派** | 几万封邮件 | **自学**：Embedding + LSTM/BERT 隐藏向量 | LSTM / BERT | ✅ 精度高、自己学语义<br>❌ 要 GPU 和大量标注数据 | `03-fasttext` / `04-bert` |
| 🚀 **LLM 派** | 0 条训练样本 | 不用特征，直接 Prompt | GPT-4 / DeepSeek | ✅ 零成本上线、能解释推理过程<br>❌ token 费用、数据要走第三方 | `05-LLM` 风格 |

> 💡 **怎么选**：看 3 个变量——**数据量、预算、可解释性需求**。
> - 数据少 + 要解释 → 走 ML 派
> - 数据多 + 要精度 → 走 DL 派
> - 没数据 + 要快 → 走 LLM 派

> 🌰 **生活类比**：判断一封邮件是不是垃圾邮件——
> - **ML 派**：训练一个老员工，手把手教他"链接超过 3 个就可疑"
> - **DL 派**：让一个新员工读 10 万封邮件，自己悟出"什么样的邮件像垃圾"
> - **LLM 派**：直接打电话问 GPT："这封邮件是垃圾邮件吗？"

### 二、核心概念依赖图（学了 A 才能学 B）

```
                          基础三件套
   ┌──────────────────────────────────────────────────┐
   │  张量 ─► Autograd ─► 损失函数 ─► 优化器 ─► 5 步训练模板   │
   └──────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
            前馈网络                      循环网络
                │                           │
                │                       RNN ─► LSTM/GRU
                │                           │
                ▼                           ▼
         注意力机制 ◄──────────────  Seq2Seq + Attention
                │
                ▼
        Transformer (Encoder ⇄ Decoder)
                │
                ▼
        BERT 预训练 ─► BERT 微调 ─► 模型压缩
                │                   (量化/剪枝/蒸馏)
                ▼
            LLM API
```

> 🌰 **生活类比**：学知识像盖楼，地基是"张量+Autograd"，柱子是"前馈+循环网络"，主体是"注意力+Transformer"，外墙装修是"BERT 微调+模型压缩"。地基不稳，上面全塌。

### 三、文档章节地图（按 README 行号导航）

| 章节 | 位置 | 层级 | 一句话总结 |
|------|------|------|-----------|
| KNN 分类算法 | [KNN 详解](#knn-classifier) | ML 基础 | "近朱者赤"——看周围 K 个邻居 |
| 数据特征处理 | [特征处理](#data-feature-processing) | ML 基础 | 归一化/标准化/独热编码 |
| 损失函数（线性回归） | [回归指标](#regression-metrics) | ML 评估 | MAE/MSE/RMSE/R² |
| **交叉熵损失** | [交叉熵](#cross-entropy) | ML 评估 | 二分类 & 多分类通吃 |
| **混淆矩阵** | [混淆矩阵](#confusion-matrix) | ML 评估 | TP/FP/FN/TN + P/R/F1/AUC |
| K-Means 聚类 | [K-Means](#kmeans) | ML 无监督 | 客户分群、肘部法则 |
| PyTorch 张量速查表 | [张量速查](#tensor-cheatsheet) | DL 基础 | 创建/形状/索引/运算 |
| **PyTorch 张量 18 例** | [张量 18 例](#tensor-18-examples) | DL 基础 | 创建→转换→运算→索引→形状→拼接 |
| **Autograd 自动微分** | [自动微分](#autograd) | DL 基础 | 计算图 + `backward()` + `zero_grad()` |
| 5 步训练模板 | [训练模板](#train-template) | DL 基础 | 前向→损失→清零→反向→更新 |
| 激活函数 | [激活函数](#activation-functions) | DL 核心 | Sigmoid/ReLU/Tanh/Softmax |
| **参数初始化** | [参数初始化](#parameter-init) | DL 核心 | 7 种初始化方式 + 选型指南 |
| **模型优化** | [模型优化](#model-optimization) | DL 核心 | 优化器演进 + 学习率 + Dropout + BN |
| 文本预处理 | [文本预处理](#text-preprocessing) | NLP 基础 | 分词/去停用词/向量化 |
| **CBOW & Skip-gram** | [Word2Vec 详解](#word2vec-cbow-skipgram) | NLP 基础 | 上下文↔中心词 + 负采样 |
| RNN/LSTM/GRU | [RNN 家族](#rnn-family) | NLP 序列模型 | 记忆细胞 + 门控机制 |
| 注意力机制四种 | [注意力机制](#attention) | NLP 进阶 | 软/硬/加性/缩放点积 |
| **Transformer 完整架构** | [Transformer](#transformer) | NLP 核心 | Encoder + Decoder + Mask |
| **编码器⇄解码器全链路** | [编解码链路](#encoder-decoder-link) | NLP 核心 | memory 纽带 + Q/K/V 来源 |
| 学习路线建议 | [学习路线](#learning-roadmap) | 路线图 | ML→DL→NLP 三阶段 |
| 文本分类实战入口 | [文本分类项目](#text-classification-project) | 项目实战 | 6 阶段递进路线 |
| 01-data 数据 EDA | [数据 EDA](#data-eda) | 实战阶段 1 | 数据清洗 + padding 选择 |
| 02-rf 随机森林 | [随机森林](#rf-section) | 实战阶段 2 | TF-IDF + RF 基线 |
| 03-fasttext | [FastText](#fasttext-section) | 实战阶段 3 | 字 vs 词级别 + autotune |
| **04-bert 微调** | [BERT 微调](#bert-section) | 实战阶段 4 | [CLS] + 三个 ID + 微调 |
| 05-LLM 大模型 | [LLM API](#llm-section) | 实战阶段 5 | DeepSeek API + Prompt 工程 |
| **06-model-compression** | [模型压缩](#compression-section) | 实战阶段 6 | 量化/剪枝/蒸馏三板斧 |
| 面试高频题精选 | [面试题](#interview-questions) | 复习 | 6 道经典题 + 答案 |

### 四、学习路径推荐（三档进阶）

| 路径 | 周期 | 学完目标 | 推荐章节顺序 |
|------|------|---------|-------------|
| 🌱 **新手** | ~30 天 | 看懂代码、能跑模型 | 环境安装 → ML 基础 → PyTorch 张量 → Autograd → 5 步模板 → 激活函数 → 参数初始化 → 模型优化(优化器+学习率) → RNN → 注意力 → Transformer 整体架构 |
| 🌿 **进阶** | ~60 天 | 能微调 BERT 完成业务分类 | 在新手基础上 + 编码器⇄解码器细节 → BERT 输入三件套 → 04-bert 全套代码 → 02-rf / 03-fasttext 对比 → Dropout + BatchNorm 调优 |
| 🌳 **专家** | ~90 天 | 模型上线 + 压缩部署 | 进阶基础上 + 蒸馏温度 T → 量化原理 → 剪枝 L1 → 上线四件套 → 05-LLM Prompt 工程 |

### 五、按需查找索引（"我想做 X" → "看 Y 章"）

| 我想… | 看哪章 |
|------|--------|
| 给老板做客户分群 | [K-Means](#kmeans) |
| 做一个垃圾邮件二分类器 | [混淆矩阵](#confusion-matrix) + [02-rf](#rf-section) |
| 做一个 10 类新闻分类器 | [04-bert](#bert-section) |
| 模型太大想压缩 | [06-model-compression](#compression-section) |
| 想理解 Transformer 怎么工作 | [编解码链路](#encoder-decoder-link) |
| 想理解 Word2Vec 怎么训练词向量 | [CBOW & Skip-gram](#word2vec-cbow-skipgram) |
| 想搞清楚 CBOW 和 Skip-gram 的区别 | [CBOW vs Skip-gram](#word2vec-cbow-skipgram) |
| 训练 loss 不下降怎么办 | [训练模板](#train-template) + [Autograd 报错](#autograd-errors) |
| 模型收敛慢 / loss 一直是 NaN | [参数初始化](#parameter-init)（检查初始化方式） |
| 不知道该用哪种初始化方法 | [参数初始化](#parameter-init)（选型指南） |
| 不知道选哪个优化器 | [模型优化](#model-optimization)（Adam 首选） |
| 训练 loss 震荡不收敛 | [模型优化](#model-optimization)（调整学习率） |
| 模型过拟合了怎么办 | [模型优化](#model-optimization)（加 Dropout） |
| 训练速度太慢 / 梯度消失 | [模型优化](#model-optimization)（加 BatchNorm） |
| 想把模型上线提供 API | [上线四件套](#deployment-pipeline) |
| 不想训练，直接用 GPT 做分类 | [05-LLM](#llm-section) |
| 评估指标怎么选 | [交叉熵](#cross-entropy) + [决策树](#metric-decision-tree) |
| 调 padding_size 时拍多少？ | [padding 选择](#padding-size-selection) |
| 蒸馏温度 T 怎么设？ | [软标签蒸馏](#temperature-t) |
| BERT 三个 ID 是啥？ | [BERT 输入](#bert-three-ids) |

### 六、关键术语索引（A-Z 速查）

| 术语 | 中文 | 链接 |
|------|------|------|
| **Attention** | 注意力机制 | [详情](#attention) |
| **Autograd** | 自动微分 | [详情](#autograd) |
| **AUC / ROC** | ROC 曲线下面积 | [详情](#auc-roc) |
| **Adam** | 自适应矩估计优化器 | [详情](#model-optimization) |
| **BCE Loss** | 二元交叉熵 | [详情](#bce-loss) |
| **BERT** | 双向编码 Transformer | [详情](#bert-section) |
| **BiLSTM** | 双向 LSTM | [详情](#bilstm) |
| **CBOW** | 连续词袋模型 | [详情](#word2vec-cbow-skipgram) |
| **`backward()`** | 反向传播触发 | [详情](#backward) |
| **BatchNorm** | 批量归一化 | [详情](#model-optimization) |
| **`[CLS]`** | 句子分类标记 | [详情](#special-tokens) |
| **Cross-Attention** | 编码-解码交叉注意力 | [详情](#encoder-decoder-link) |
| **Cross Entropy** | 交叉熵损失 | [详情](#cross-entropy) |
| **Decoder** | 解码器 | [详情](#decoder) |
| **Distillation** | 知识蒸馏 | [详情](#distillation) |
| **Dropout** | 随机失活正则化 | [详情](#model-optimization) |
| **Encoder** | 编码器 | [详情](#encoder) |
| **EWMA** | 指数加权平均 | [详情](#model-optimization) |
| **F1 Score** | F1 分数 | [详情](#precision-recall-f1) |
| **FastText** | 浅层快速分类 | [详情](#fasttext-section) |
| **GRU** | 门控循环单元 | [详情](#gru) |
| **He Initialization** | Kaiming 初始化 | [详情](#parameter-init) |
| **K-Means** | K 均值聚类 | [详情](#kmeans) |
| **Kaiming 初始化** | He 初始化 | [详情](#parameter-init) |
| **KNN** | K 近邻 | [详情](#knn-classifier) |
| **LayerNorm** | 层归一化 | [详情](#encoder) |
| **Learning Rate** | 学习率 | [详情](#model-optimization) |
| **LSTM** | 长短时记忆 | [详情](#lstm) |
| **mask** | 掩码 | [详情](#mask) |
| **memory** | 编码器输出 | [详情](#encoder-decoder-link) |
| **Momentum** | 动量法优化器 | [详情](#model-optimization) |
| **Multi-Head Attention** | 多头注意力 | [详情](#multi-head-attention) |
| **NSP** | 下一句预测（BERT 任务）| [详情](#bert-pretraining-tasks) |
| **Optimizer** | 优化器 | [详情](#model-optimization) |
| **Padding Size** | 序列填充长度 | [详情](#padding-size-selection) |
| **Positional Encoding** | 位置编码 | [详情](#positional-encoding) |
| **Precision / Recall** | 精确率 / 召回率 | [详情](#precision-recall-f1) |
| **Pruning** | 剪枝 | [详情](#pruning) |
| **Quantization** | 量化 | [详情](#quantization) |
| **`requires_grad`** | 梯度跟踪标记 | [详情](#requires-grad) |
| **RMSprop** | RMSprop 优化器 | [详情](#model-optimization) |
| **RNN** | 循环神经网络 | [详情](#rnn) |
| **Self-Attention** | 自注意力 | [详情](#encoder) |
| **SGD** | 随机梯度下降 | [详情](#model-optimization) |
| **Skip-gram** | 跳元模型 | [详情](#word2vec-cbow-skipgram) |
| **Softmax** | 多分类输出层 | [详情](#softmax) |
| **Teacher Forcing** | 教师强制（训练）| [详情](#encoder-decoder-link) |
| **TF-IDF** | 词频-逆文档频率 | [详情](#tf-idf) |
| **Transformer** | Transformer 架构 | [详情](#transformer) |
| **Word2Vec** | 词到向量模型 | [详情](#word2vec-cbow-skipgram) |
| **Xavier 初始化** | Glorot 初始化 | [详情](#parameter-init) |
| **`zero_grad()`** | 梯度清零 | [详情](#zero-grad) |

> 💡 **使用建议**：
> - **第一次学**：按"四"的新手路径走，**严禁跳级**
> - **复习**：先看"二"的依赖图回忆全貌，再跳到具体章节细读
> - **解决问题**：直接查"五"的反向索引或"六"的术语表
> - **面试前**：从顶到尾过一遍"三"的章节地图，重点章节加粗

### 七、常见术语速释

> <small>以下术语在文档中反复出现，本表给出一句话定义，方便快速查阅。（加粗项 = 面试高频考点）</small>
>
> | 术语 | 一句话解释 |
> |------|-----------|
> | **鲁棒性** | 模型对异常值、噪声、数据波动的**抵抗能力**——越鲁棒，越不容易被个别坏数据带偏 |
> | **梯度消失** | 反向传播时梯度逐层衰减到接近 0，导致**前面几层学不动**，常见于 Sigmoid/Tanh + 深层网络 |
> | **梯度爆炸** | 反向传播时梯度指数级增大，loss 变成 **NaN**，训练直接崩溃 |
> | **过拟合** | 模型把训练数据的**噪声都记下来了**，训练集表现好但测试集差，像"背答案没学懂" |
> | **欠拟合** | 模型连训练数据的**基本规律都没学到**，训练集和测试集表现都差，像"没认真听课" |
> | **收敛** | loss 不再明显下降、参数趋于稳定，说明模型**训练到位了** |
> | **泛化** | 模型在**没见过的数据**上的表现能力——泛化好才是真的好 |
> | **正则化** | 防止过拟合的一类技术统称（Dropout、L2 权重衰减等），核心思想是**给模型加约束** |
> | **互斥**（多分类） | 每个样本**只能属于一个类别**（如"体育"就不能是"财经"），用 Softmax + CrossEntropyLoss |
> | **One-hot 编码** | 用 N 位 0/1 表示 N 个类别，如 3 类 → `[1,0,0]` `[0,1,0]` `[0,0,1]`，**只有一位是 1** |

---

## 目录结构

```
AI-Learning/
├── Machine-Learnning/          # 机器学习
│   ├── Supervised Learning/    # 监督学习
│   │   ├── Classification task/     # 分类任务
│   │   ├── Regression task/         # 回归任务
│   │   └── 数据归一化和标准化/       # 特征工程
│   └── UnSupervised Learning/  # 无监督学习
├── Deep learnning/             # 深度学习
│   ├── 01-pytorch框架各种api示例/   # PyTorch基础
│   ├── 02-神经网络/                 # 神经网络基础（含自动微分）
│   ├── 03-激活函数/                 # 激活函数
│   ├── 04-参数初始化/               # 参数初始化
│   ├── 04_损失函数/                 # 损失函数
│   ├── 05_模型优化/                 # 模型优化（优化器/学习率/Dropout/BN）
│   └── 06_综合案例_手机价格分类预测/ # DL 综合实战
├── NLP/                        # 自然语言处理
├── 文本分类项目/                # 文本分类实战 (THUCNews 10分类)
│   ├── 01-data/                # 数据 + EDA
│   ├── 02-rf/                  # 随机森林 + TF-IDF
│   ├── 03-fasttext/            # FastText
│   ├── 04-bert/                # BERT 微调
│   ├── 05-LLM/                 # DeepSeek API
│   └── 06-model-compression/   # 量化/剪枝/蒸馏
├── pdf/                        # 学习资料
└── README.md                   # 项目说明
```

## 环境安装

```shell
pip install scikit-learn torch numpy pandas matplotlib jieba
```

## 人工智能发展的三要素

> AI 不是黑科技，是"数据 + 算法 + 算力"三者堆出来的。三者缺一不可，但重要性不等价——下表帮你建立直觉。

| 要素 | 说明 | 重要性 |
|------|------|--------|
| **数据** | 决定了模型最终效果的上限 | 🌟🌟🌟🌟🌟 |
| **算法** | 解决问题的思路/方法 | 🌟🌟🌟🌟 |
| **算力** | CPU/GPU/TPU等计算资源 | 🌟🌟🌟 |

> 💡 **核心理念**: 数据质量 > 数据数量 > 算法优化 > 算力提升

## 算法的学习方式

> 按"训练数据有没有标签"，机器学习被划成三大流派。本节先用一张对照表帮你区分清楚。

| 区别 |     监督学习      | 无监督学习 | 半监督学习 |
| :--: | :---------------: | :--------: | :--------: |
| 特征 |         ✅️         |     ✅️      |     ✅️      |
| 标签 |         ✅️         |     ❌️      |     部分✅️  |
| 任务 | 分类任务/回归任务 |  聚类任务  |  混合任务  |
| 典型算法 | KNN、SVM、决策树 | K-Means、PCA | 自训练、协同训练 |
| 应用场景 | 垃圾邮件检测、房价预测 | 客户分群、异常检测 | 医疗诊断、图像标注 |

### 有监督学习

| 算法任务 | 标签类型 |   案例   | 常用算法 | 评估指标 |
| :------: | :------: | :------: | :------: | :------: |
| 分类算法 |   离散   | 垃圾邮件、疾病诊断 | KNN、SVM、决策树、随机森林 | 准确率、精确率、召回率、F1 |
| 回归算法 |   连续   | 房价预测、股票价格 | 线性回归、岭回归、Lasso | MAE、MSE、RMSE、R² |

<a id="knn-classifier"></a>
#### 分类算法详解 - KNN (K-Nearest Neighbors)

**核心思想**: 根据最近的K个邻居的类别进行投票决定当前样本的类别

**优点**:
- ✅ 简单直观，易于理解
- ✅ 无需训练过程（懒惰学习）
- ✅ 对异常值不敏感

**缺点**:
- ❌ 计算复杂度高（需计算所有距离）
- ❌ 需要存储全部训练数据
- ❌ 对特征尺度敏感（需标准化）

**关键参数**:
- `n_neighbors`: K值选择（通常3-7）
- `weights`: 权重方式（uniform/distance）
- `metric`: 距离度量（euclidean/manhattan/minkowski）

<a id="data-feature-processing"></a>
### 数据特征处理

数据特征处理是机器学习中至关重要的预处理步骤，主要包括**归一化**和**标准化**两种方法。

> <small>**为什么要做？** 不同特征的量纲差距（如"年龄 0~100" vs "收入 0~100000"）会导致：① 梯度下降震荡不收敛；② 距离类算法（KNN/SVM）被大数值特征主导。统一尺度后模型收敛更快、更稳。</small>

> ⚠️ **注意**: 归一化容易受异常值影响，在实际环境中基本都使用标准化

| 对比维度 | 归一化 (Min-Max) | 标准化 (Z-Score) |
|---------|--------|--------|
| **目标范围** | [0, 1] 或 [-1, 1] | 均值为 0，方差为 1 |
| **分布影响** | 不改变分布形状 | 转化为标准正态分布 |
| **异常值敏感度** | **非常敏感** | 相对不敏感 |
| **计算依赖** | 最大值、最小值 | 均值、标准差 |
| **适用数据** | 界限明显的数据 | 近似正态分布的数据 |
| **典型算法** | 神经网络、KNN、图像处理 | 线性模型、SVM、PCA、逻辑回归 |
| **公式** | `(x-min)/(max-min)` | `(x-mean)/std` |

#### 选择建议

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 数据分布未知 | 标准化 | 更稳健，不受边界限制 |
| 图像像素值处理 | 归一化 | 天然有界 [0,255] |
| 神经网络输入 | 标准化 | 加速收敛，避免梯度消失 |
| KNN/SVM | 标准化 | 距离计算更合理 |
| 存在异常值 | 标准化 | 对异常值鲁棒性更好 |

<a id="regression-metrics"></a>
### (损失函数)线性回归模型评估指标

| 指标 | 中文名称 | 公式 | 特点 | 适用场景 |
|------|---------|------|------|----------|
| **MAE** | 平均绝对误差 | `1/n * Σ\|y_true - y_pred\|` | 对异常值稳健，解释性强 | 异常值较多的数据 |
| **MSE** | 均方误差 | `1/n * Σ(y_true - y_pred)²` | 对异常值敏感，便于梯度下降 | 需要惩罚大误差的场景 |
| **RMSE** | 均方根误差 | `√MSE` | 与原始数据量纲一致，更直观 | 结果需要可解释性 |
| **R² Score** | 决定系数 | `1 - SS_res/SS_tot` | 表示模型解释方差的比例 | 比较不同模型的拟合优度 |

**选择建议**:
- 🎯 **首选**: RMSE（直观）+ R²（相对性能）
- 🔍 **调试**: MAE（定位问题）+ MSE（优化目标）

<a id="cross-entropy"></a>
### (损失函数)交叉熵损失（Cross Entropy Loss）—— 二分类 & 多分类通吃

> ❗ **常见误解澄清**：交叉熵 **不只用于二分类**。它对二分类、多分类、多标签分类**都通用**，二分类只是它的一个特例。本仓库的 BERT 文本分类（10 类）、蒸馏学生模型，全部使用 `nn.CrossEntropyLoss`。

#### 一、核心思想（一句话）

**衡量"预测概率分布"和"真实分布"之间的差距**——预测越接近真实，损失越小；越偏离，损失越大。

#### 二、三种场景对照表

| 场景 | 标签形式 | 输出层 | 损失公式 | PyTorch API |
|------|---------|--------|----------|-------------|
| **二分类**（猫 vs 狗） | 0 或 1 | Sigmoid → 1 个概率 | `L = -[y·log(p) + (1-y)·log(1-p)]` | <a id="bce-loss"></a>`nn.BCELoss` 或 `nn.BCEWithLogitsLoss`（推荐） |
| **多分类**（10 个新闻类别，互斥） | 0~9 之一 | Softmax → C 个概率 | `L = -Σᵢ yᵢ·log(pᵢ)`（yᵢ 是 one-hot） | **`nn.CrossEntropyLoss`**（最常用） |
| **多标签**（一篇文章可同时是"科技"+"财经"） | <a id="softmax"></a>多 hot 向量 | 每个类别独立 Sigmoid | 每类单独算 BCE，再求和/平均 | `nn.BCEWithLogitsLoss` |

#### 三、两个公式的关系（看清"通用性"）

**多分类公式：** `L = -Σᵢ₌₁ᶜ yᵢ·log(pᵢ)`

当类别数 `C=2`、且 `y=[1-y_true, y_true]`、`p=[1-p, p]` 时，把求和展开：

```
L = -[(1-y)·log(1-p) + y·log(p)]
```

→ 这就是二分类公式。所以**二分类是多分类的特例**，并不是另一种损失。

> 🌰 **生活类比**：多分类公式像"通用螺丝刀"，二分类公式像"专卖二号头"——本质同一把刀，只是换了头。

#### 四、PyTorch 四大 API 区别（最容易混淆）

| API | 输入 logits 是否需要先 softmax/sigmoid | 标签格式 | 适用 |
|------|-----------------------------|---------|------|
| `nn.CrossEntropyLoss` | **不要**（内部已含 log_softmax） | 类别索引 `LongTensor[batch]`，如 `[0,3,7]` | **多分类首选** |
| `nn.NLLLoss` | **要**（必须先经 `log_softmax`） | 同上 | 模型自己输出 `log_softmax` 时用 |
| `nn.BCELoss` | **要**（必须先经 `sigmoid`） | `FloatTensor[batch]` 或 `[batch,C]`，0~1 | 二分类 / 多标签 |
| `nn.BCEWithLogitsLoss` | **不要**（内部已含 sigmoid） | 同上 | 推荐替代 `BCELoss`（数值更稳） |

**关键等价关系**（务必记住）：

```
nn.CrossEntropyLoss   = log_softmax + nn.NLLLoss
nn.BCEWithLogitsLoss  = sigmoid     + nn.BCELoss
```

> 🌰 **生活类比**：
> - 带 `WithLogits` / `CrossEntropy` 版本 = "免洗"洗发水（直接抹原始 logits）
> - 不带的版本 = 普通洗发水（先打湿，再用）
> - 直接喂 logits 就选"免洗"版本——更省事，数值更稳定

#### 五、最常见的代码模板（多分类）

```python
import torch
import torch.nn as nn

# 模型输出原始 logits（不要自己 softmax！）
logits = model(input_ids)              # [batch, num_classes]
labels = torch.tensor([2, 5, 1, 9])    # [batch] 类别索引，必须 LongTensor

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)         # 内部自动 log_softmax + NLLLoss
loss.backward()
```

> ⚠️ **常见报错**：
> - 标签传 `float`：报 `expected scalar type Long`，必须 `.long()`
> - logits 自己先做了 `softmax`：训练 loss 不下降，因为做了两次归一化

#### 七、本仓库实战示例

| 文件 | 任务 | 损失函数 |
|------|------|----------|
| [文本分类项目/04-bert/](文本分类项目/04-bert/) | 10 类新闻分类（多分类） | `nn.CrossEntropyLoss()` |
| [hard_label_distillation.py](文本分类项目/06-model-compression/bert_distll/hard_label_distillation.py) | 硬标签蒸馏 | `nn.CrossEntropyLoss()(student_logits, teacher_preds)` |
| [soft_label_distillation.py](文本分类项目/06-model-compression/bert_distll/soft_label_distillation.py) | 软标签蒸馏 | `KLDivLoss + CrossEntropyLoss` 加权（见后文） |

#### 七、面试速答 / 易错点

1. **Q：交叉熵只能用于二分类吗？**
   A：**不是**。多分类（`CrossEntropyLoss`）、多标签（`BCEWithLogitsLoss` 逐类）都用交叉熵思想，二分类只是 `C=2` 的特例。

2. **Q：为什么分类用交叉熵不用 MSE？**
   A：MSE 配合 sigmoid/softmax 时**梯度容易消失**（误差大时梯度反而趋零），而交叉熵的梯度形式很干净（`p - y`），训练更稳。

3. **Q：`nn.CrossEntropyLoss` 的 logits 要不要自己 softmax？**
   A：**绝对不要**！它内部已经 `log_softmax`，自己再做一次会导致数值不稳和梯度异常。

4. **Q：二分类用 `CrossEntropyLoss` 还是 `BCEWithLogitsLoss`？**
   A：都行。`CrossEntropyLoss` 输出 2 个 logits 走 softmax；`BCEWithLogitsLoss` 输出 1 个 logit 走 sigmoid。**多标签必须用 BCE**（因为各类别不互斥）。

5. **Q：`label_smoothing` 是什么？**
   A：把 one-hot `[0,0,1,0]` 软化成 `[0.025, 0.025, 0.925, 0.025]`，缓解过拟合。`nn.CrossEntropyLoss(label_smoothing=0.1)` 一行启用。

---

<a id="confusion-matrix"></a>
### (损失函数)二分类模型评估指标 —— 混淆矩阵

#### 一、混淆矩阵长什么样（先看图）

| 预测\实际 | 正例 (Positive) | 负例 (Negative) |
|-----------|----------------|----------------|
| **正例**  | TP (真阳性)     | FP (假阳性/误报) |
| **负例**  | FN (假阴性/漏报) | TN (真阴性)     |

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_true, y_pred)
# 返回 shape=(2,2) 的 ndarray:
# [[TN, FP],
#  [FN, TP]]
```

**4 个格子的助记口诀**：
- **第一个字母**：T/F = 预测**对没对**（True 对、False 错）
- **第二个字母**：P/N = 预测**说是正还是负**（Positive 正、Negative 负）

> 🌰 **生活类比 · 新冠核酸检测**：
> - **TP**：阳性病人 → 检出阳性 ✅（确实有病，也查出来了）
> - **FP**：健康人 → 检出阳性 ❌（误报，好人被冤枉）
> - **FN**：阳性病人 → 检出阴性 ❌（漏报，传染源放走了，最危险！）
> - **TN**：健康人 → 检出阴性 ✅（确实没病，也没冤枉）

<a id="precision-recall-f1"></a>
#### 二、衍生指标（5 个最常用）

| 指标 | 公式 | 含义 | 适用场景 | sklearn API |
|------|------|------|----------|-------------|
| **准确率 (Accuracy)** | `(TP+TN)/(TP+TN+FP+FN)` | 整体预测正确的比例 | 类别平衡的数据集 | `accuracy_score(y_true, y_pred)` |
| **精确率 (Precision)** | `TP/(TP+FP)` | **预测为正**的里头有多少真的是正 | 关注误报成本（如垃圾邮件） | `precision_score(y_true, y_pred, average='binary')` |
| **召回率 (Recall)** | `TP/(TP+FN)` | **真正是正**的里头有多少被找出来 | 关注漏报成本（如疾病检测） | `recall_score(y_true, y_pred, average='binary')` |
| **F1 分数** | `2·P·R/(P+R)` | 精确率和召回率的调和平均 | 需要兼顾两者 | `f1_score(y_true, y_pred, average='binary')` |
| **特异度 (Specificity)** | `TN/(TN+FP)` | 真正是负的里头有多少被识别 | 医学检测、ROC 曲线 | `recall_score(y_true, y_pred, pos_label=0)` |

> 🌰 **精确率 vs 召回率怎么记**：
> - **精确率 = "宁缺勿滥"**：我说是正的就一定是正的（垃圾邮件：宁可漏几个垃圾，不能把正常邮件丢进垃圾箱）
> - **召回率 = "宁滥勿缺"**：所有正的我都要找出来（癌症筛查：宁可吓几个健康人去复查，不能放过一个真病人）
> - **F1 = "我都要"**：当两者都重要时取调和平均（调和平均的特点：两个值差距大时拉低，逼你两手都硬）

#### 三、举个具体数字感受一下

> 假设 100 封邮件里 20 封真垃圾邮件，模型预测：

| 预测\实际 | 真垃圾 (20) | 真正常 (80) |
|----------|-------------|-------------|
| 预测垃圾  | TP=15       | FP=10       |
| 预测正常  | FN=5        | TN=70       |

- **Accuracy** = (15+70)/100 = **85%**
- **Precision** = 15/(15+10) = **60%**（预测的 25 封垃圾里只有 15 封真垃圾）
- **Recall** = 15/(15+5) = **75%**（20 封真垃圾找回了 15 封）
- **F1** = 2·0.6·0.75/(0.6+0.75) = **0.667**
- **Specificity** = 70/(70+10) = **87.5%**

> 💡 **类别不平衡警告**：如果模型偷懒，把所有邮件都判为"正常"，准确率仍有 80%，但 Recall = 0%。**只看准确率会被骗！**

<a id="auc-roc"></a>
#### 四、ROC 曲线 & AUC（用一句话理解）

**AUC 的直观含义**：随机抽**一个垃圾邮件**和**一封正常邮件**，模型给垃圾邮件打分**高于**正常邮件的概率。

| AUC 值 | 模型水平 | 通俗解释 |
|--------|---------|---------|
| 1.0    | 完美 | 永远能把"真垃圾"排在"真正常"前面 |
| 0.9    | 优秀 | 90% 情况下排得对 |
| 0.7    | 一般 | 70% 情况下排得对 |
| 0.5    | 瞎猜 | 等于抛硬币 |

> 🌰 **生活类比**：把所有邮件按"垃圾概率"从高到低排队。AUC 越高，真垃圾就越集中在队伍前面。

> 💡 **为什么 AUC 在类别不平衡下更可靠**？
> 因为 AUC 看的是"排序能力"，不依赖具体阈值。Accuracy 在 99:1 不平衡下全猜负就有 99%，但 AUC 不会被骗。

#### 五、sklearn 一行 API 速查

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

y_true = [1, 0, 1, 1, 0, 0, 1, 0]   # 真实标签
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]   # 模型预测

# 1. 看混淆矩阵 4 个格子
print(confusion_matrix(y_true, y_pred))
# [[3 1]    ← TN=3, FP=1
#  [1 3]]   ← FN=1, TP=3

# 2. 一键打印 Precision / Recall / F1
print(classification_report(y_true, y_pred, target_names=['正常', '垃圾']))

# 3. 单独算
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)

# 4. AUC（注意第二个参数是"概率"，不是 0/1 预测）
y_pred_proba = [0.9, 0.1, 0.8, 0.4, 0.2, 0.6, 0.95, 0.3]
auc = roc_auc_score(y_true, y_pred_proba)
```

> ⚠️ **常见报错**：`roc_auc_score` 第二个参数必须是**概率**（`predict_proba()[:,1]`），不是 `predict()` 给的 0/1 标签。

<a id="metric-decision-tree"></a>
#### 六、选择指标的决策树（二分类版）

```
你的任务关心什么？
├─ 整体对就行 + 类别均衡 → Accuracy
├─ 怕漏报（医疗、风控、欺诈） → 优先 Recall
├─ 怕误报（垃圾邮件、推荐） → 优先 Precision
├─ 二者都重要 → F1 分数
└─ 类别严重不平衡 → AUC（不依赖阈值）
```

> 💡 **重要提示**: 
> - 类别不平衡时，**不要只看准确率**！
> - 医疗诊断：优先保证高 **Recall**（宁可误报，不可漏报）
> - 垃圾邮件：优先保证高 **Precision**（宁可漏报，不可误报）
> - 想综合评估模型 → 用 **AUC** + **F1**

### 无监督学习

| 算法 | 核心思想 | 适用场景 | 优缺点 |
|------|---------|---------|--------|
| **K-Means聚类** | 迭代更新簇中心，最小化簇内距离 | 客户分群、图像分割 | ✅ 简单高效 ❌ 需指定K值，对异常值敏感 |
| **层次聚类** | 构建树状结构，逐步合并或分裂簇 | 小数据集、生物信息学 | ✅ 无需指定K值 ❌ 计算复杂度高 |
| **DBSCAN** | 基于密度的聚类，识别任意形状簇 | 噪声数据、空间聚类 | ✅ 自动确定簇数，抗噪声 ❌ 参数敏感 |
| **PCA降维** | 线性变换，保留最大方差方向 | 数据可视化、特征压缩 | ✅ 去相关，降噪 ❌ 仅线性关系 |

<a id="kmeans"></a>
#### K-Means聚类算法详解

**算法流程**:
1. 随机初始化K个簇中心
2. 将每个样本分配到最近的簇中心
3. 重新计算每个簇的中心（均值）
4. 重复步骤2-3直到收敛

**关键参数**:
- `n_clusters`: 簇的数量K（需预先指定）
- `init`: 初始化方法（'k-means++'推荐）
- `max_iter`: 最大迭代次数
- `n_init`: 运行次数，选择最优结果

**优点**:
- ✅ 算法简单，易于实现
- ✅ 时间复杂度低 O(n*K*t)
- ✅ 适用于大规模数据集

**缺点**:
- ❌ 需要预先指定K值
- ❌ 对初始中心点敏感
- ❌ 只能发现球状簇
- ❌ 对异常值敏感

**K值选择方法**:
- 📊 **肘部法则 (Elbow Method)**: 观察SSE随K变化的拐点
- 📈 **轮廓系数 (Silhouette Score)**: 衡量簇内紧密度和簇间分离度



# 深度学习

> **深度学习 = 多层神经网络 + 大量数据 + GPU 算力**。本章从 PyTorch 张量基础出发，依次过自动微分（Autograd）、训练标准模板、激活函数四件套，是理解 NLP/CV 等所有上层任务的地基。三句话掌握脉络：
> 1. **张量** 是数据的统一表示（标量/向量/矩阵/高维全用 `torch.Tensor`）
> 2. **Autograd** 替你自动算梯度，无需手推链式法则
> 3. **5 步训练模板**（前向→损失→清零→反向→更新）适配所有模型，从 MLP 到 BERT 都一样

## PyTorch基础

> PyTorch 入门第一站：张量。把它当成"加强版 numpy"——会自动求导、能跑 GPU。本节先过常用 API 速查表，再看 18 个示例文件细节。

<a id="tensor-cheatsheet"></a>
### 张量(Tensor)操作速查表

| 操作类型 | 方法 | 示例 | 说明 |
|---------|------|------|------|
| **创建张量** | `torch.tensor()` | `torch.tensor([1,2,3])` | 从列表创建 |
| | `torch.zeros()` | `torch.zeros(3,4)` | 全0张量 |
| | `torch.ones()` | `torch.ones(3,4)` | 全1张量 |
| | `torch.randn()` | `torch.randn(3,4)` | 标准正态分布 |
| | `torch.arange()` | `torch.arange(0,10,2)` | 等差数列 |
| **形状操作** | `.shape` / `.size()` | `x.shape` | 获取形状 |
| | `.reshape()` | `x.reshape(2,-1)` | 改变形状 |
| | `.view()` | `x.view(2,-1)` | 视图重塑 |
| | `.unsqueeze()` | `x.unsqueeze(0)` | 增加维度 |
| | `.squeeze()` | `x.squeeze()` | 减少维度 |
| **数据类型** | `.dtype` | `x.dtype` | 查看数据类型 |
| | `.float()` | `x.float()` | 转为浮点型 |
| | `.int()` | `x.int()` | 转为整型 |
| **运算操作** | `+,-,*,/` | `x + y` |  element-wise运算 |
| | `torch.mm()` | `torch.mm(A,B)` | 矩阵乘法 |
| | `torch.bmm()` | `torch.bmm(A,B)` | 批量矩阵乘法 |
| | `.sum()` | `x.sum(dim=1)` | 求和 |
| | `.mean()` | `x.mean()` | 平均值 |
| **索引切片** | `x[0]` | `x[0]` | 第一行 |
| | `x[:,1]` | `x[:,1]` | 第二列 |
| | `x[0:2,1:3]` | `x[0:2,1:3]` | 子矩阵 |
| **设备转移** | `.to('cuda')` | `x.to('cuda')` | 转到GPU |
| | `.cpu()` | `x.cpu()` | 转到CPU |

> 💡 **提示**: 按住Ctrl+鼠标左键点击函数可查看源码

<a id="tensor-18-examples"></a>
### PyTorch 张量 API 详细教程（18 个示例文件）

> 📂 文件来源：[Deep learnning/01-pytorch框架各种api示例/](Deep%20learnning/01-pytorch框架各种api示例/) 共 18 个 `.py` 文件，每个聚焦一个主题。下面按"创建 → 转换 → 运算 → 索引 → 形状 → 拼接"的脉络梳理。

#### 1️⃣ 创建张量七大入口

| 方法 | 示例 | 含义 | 生活化类比 |
|------|------|------|----------|
| `torch.tensor()` | `torch.tensor([[1,2],[3,4]])` | 从 Python 列表/numpy 创建 | 自己手写一份食谱 |
| `torch.IntTensor / FloatTensor` | `torch.IntTensor(2,3)` | 指定数据类型创建 | 选不锈钢锅还是铁锅 |
| `torch.arange(0,10,2)` | → `[0,2,4,6,8]` | 等差数列（不含右端） | 楼层 1、3、5、7 |
| `torch.linspace(0,10,5)` | → `[0,2.5,5,7.5,10]` | 等分区间（含两端） | 把一段绳子等分 5 段 |
| `torch.rand(2,3)` | 0~1 均匀分布 | 公平随机 | 抽签 |
| `torch.randn(2,3)` | 标准正态 | 钟形随机 | 全班身高分布 |
| `torch.randint(0,10,(2,3))` | 整数随机 | 摇骰子 | 抽 1~10 号 |
| `torch.zeros / ones / full` | `torch.full((2,3),8)` | 全 0/1/指定值 | 全空教室 / 全满座 |
| `torch.zeros_like(x)` | 同 `x` 形状的全 0 | "和它一样大" 模板 | 同款空盒子 |

#### 2️⃣ 数据类型与互转（[文件05](Deep%20learnning/01-pytorch框架各种api示例/05_张量的数据类型转换.py) ~ [文件07](Deep%20learnning/01-pytorch框架各种api示例/07_张量和标量互转.py)）

| 操作 | 示例 | 用途 |
|------|------|------|
| 看类型 | `tensor.dtype` | 查身份证 |
| 转浮点 | `tensor.float()` / `.type(torch.float32)` | 神经网络默认要 float |
| 转整型 | `tensor.int()` / `.long()` | 索引和标签必须 long |
| numpy → tensor | `torch.from_numpy(arr)` | **共享内存**（一改全改） |
| tensor → numpy | `tensor.numpy()` | 同样共享内存 |
| 单元素 → 标量 | `tensor.item()` | 损失值打印必备 |

> ⚠️ **坑点**：`torch.from_numpy()` 和 `.numpy()` 都是"借东西"不是"复制"，原地修改会互相污染。要彻底分开用 `.numpy().copy()`。

#### 3️⃣ 数学运算与聚合（[文件08](Deep%20learnning/01-pytorch框架各种api示例/08_张量的加减乘除负号基本运算.py) ~ [文件10](Deep%20learnning/01-pytorch框架各种api示例/10_张量的其他运算函数.py)）

| 类型 | 方法 | 说明 |
|------|------|------|
| 元素级运算 | `+ - * /` 或 `torch.add/sub/mul/div` | 同形状逐位运算 |
| **inplace 运算** | `tensor.add_(2)` | **带下划线 = 原地修改** |
| 矩阵乘法 | `A @ B` 或 `torch.matmul(A, B)` | 二维及以上 |
| 二维矩阵乘 | `torch.mm(A, B)` | 仅二维 |
| 批量矩阵乘 | `torch.bmm(A, B)` | (b,m,n)×(b,n,k) |
| 向量点积 | `torch.dot(a, b)` | 一维专用 |
| 聚合函数 | `.sum() .mean() .max() .min()` | 求和/平均/极值 |
| 数学函数 | `.pow(2) .sqrt() .exp() .log()` | 幂/开根/指数/对数 |

> 🌰 **生活类比**：`add_()` 像在原本的笔记上直接涂改；`add()` 像复印一份再改。

#### 4️⃣ 索引切片（[文件11](Deep%20learnning/01-pytorch框架各种api示例/11_张量的基础索引操作.py) ~ [文件12](Deep%20learnning/01-pytorch框架各种api示例/12_张量的多维索引.py)）

```python
x[0]              # 第 0 行
x[:, 1]           # 第 1 列（所有行）
x[0:2, 1:3]       # 子矩阵
x[x > 5]          # 布尔索引：取所有 > 5 的元素
x[[0, 2], [1, 3]] # 高级索引：取 (0,1) 和 (2,3) 两个元素
```

> 🌰 **生活类比**：
> - 基础切片 = 在 Excel 里框选一块区域
> - 布尔索引 = 筛选成绩 > 60 的同学
> - 高级索引 = 按学号清单一个个挑出来

#### 5️⃣ 形状变换（[文件13](Deep%20learnning/01-pytorch框架各种api示例/13_张量获取形状和修改形状.py) ~ [文件17](Deep%20learnning/01-pytorch框架各种api示例/17_张量的是否连续判断以及修改操作.py)）

| 方法 | 作用 | 生活类比 |
|------|------|---------|
| `.shape` / `.size()` | 看形状 | 量身高体重 |
| `.reshape(2,-1)` | 改形状（自动计算 -1） | 把长面条卷成饼 |
| `.view(2,-1)` | 同 reshape，但要求**内存连续** | 同上，但只能在原料连续时用 |
| `.unsqueeze(0)` | 在第 0 维加一维 | 给煎饼套个塑料袋（多一层） |
| `.squeeze()` | 去掉所有为 1 的维 | 拆掉空塑料袋 |
| `[None, :]` | 等价于 `unsqueeze(0)` | 简洁写法 |
| `.transpose(0, 1)` | 交换两维 | 表格行列对调 |
| `.permute(2, 0, 1)` | 任意维度重排 | 三人换座位 |
| `.is_contiguous()` | 判断内存连续 | 检查面条有没有断 |
| `.contiguous()` | 重新整理成连续 | 把断面条粘起来 |

> ⚠️ **常见报错**：`.view()` 在非连续张量上会失败 → 先 `.contiguous().view()` 或直接用 `.reshape()`。

#### 6️⃣ 拼接合并（[文件18](Deep%20learnning/01-pytorch框架各种api示例/18_张量的拼接操作.py)）

```python
# cat: 已有维度上拼接（不增加新维度）
torch.cat([t1, t2], dim=0)   # (2,3)+(2,3) → (4,3)
torch.cat([t1, t2], dim=1)   # (2,3)+(2,3) → (2,6)

# stack: 新增一个维度再叠
torch.stack([t1, t2])        # (2,3)+(2,3) → (2,2,3)
```

> 🌰 **生活类比**：
> - `cat` = 把两摞煎饼接成一摞更长的（数量增加）
> - `stack` = 把两摞煎饼叠成两层（多了一个"层"维度）

#### 一句话记忆口诀

> **"创建用 zeros/randn，转型用 .float()，运算 @ 走一遍，索引切片要熟练，形状靠 reshape，拼接看 cat/stack。"**

<a id="autograd"></a>
## 自动微分（Autograd）从零理解

> 📂 文件来源：[Deep learnning/02-神经网络/自动微分/](Deep%20learnning/02-神经网络/自动微分/)

### 核心思想

**自动微分**就是 PyTorch 自动帮你算"梯度"。
你只要写好"正向计算"，PyTorch 会偷偷记下来怎么算的，反向时自动求导。

> 🌰 **生活类比**：去超市买菜结账，超市每过一道收银台就会打一张"小票链条"（计算图）。等你要退货时（反向传播），按这条链条原路返回，每一站告诉你这一步贡献了多少钱（梯度）。

### 计算图三要素

| 要素 | 说明 |
|------|------|
| **叶子节点 (Leaf)** | 你创建的、`requires_grad=True` 的张量（如 w、b） |
| **中间节点** | 由叶子节点经过运算得到的（如 `z = x @ w + b`） |
| **`.grad_fn`** | 每个非叶子张量都有，记录"它是怎么算来的" |

### 三步标准模板（[01_单轮.py](Deep%20learnning/02-神经网络/自动微分/01_自动微分_更新权重_单轮.py)）

```python
# 1. 启用梯度跟踪
w = torch.tensor(10.0, requires_grad=True)

# 2. 正向计算 + 计算损失
loss = (w - 5) ** 2

# 3. 反向传播，计算梯度
loss.backward()
print(w.grad)  # → tensor(10.) 表示 dloss/dw = 10
```

<a id="backward"></a>
### `backward()` 必须对标量调用

```python
y = x ** 2          # y 是向量
y.backward()        # ❌ 报错！必须是标量
y.sum().backward()  # ✅ 标量
```

> 🌰 **生活类比**：
> - 反向传播是"算总分对每道题的依赖"，必须有一个"总分"才能反推。
> - 向量没有"总分"概念，所以要 `.sum()` 加起来变成标量。

<a id="zero-grad"></a>
### 多轮训练的"梯度清零"陷阱（[02_多轮.py](Deep%20learnning/02-神经网络/自动微分/02_自动微分_更新权重_多轮.py)）

```python
for epoch in range(100):
    loss = (w - 5) ** 2
    loss.backward()           # ⚠️ 梯度会累加！
    
    with torch.no_grad():
        w -= 0.1 * w.grad     # 手动更新
    
    w.grad.zero_()            # 必须清零，否则下轮梯度叠加上来
```

> 🌰 **生活类比**：体重秤每天用前要清零，不然今天 70 kg 第二天就显示 140 kg；梯度也一样，不清零会越积越大，模型直接爆炸。

### 全连接 z = x @ w + b 的求导（[03_全连接.py](Deep%20learnning/02-神经网络/自动微分/03_自动微分_整体应用_推导wb梯度.py)）

```python
x = torch.tensor([[1., 2.]])
w = torch.randn(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
y_true = torch.tensor([[5.]])

# 正向
z = x @ w + b
loss = ((z - y_true) ** 2).mean()

# 反向
loss.backward()
print(w.grad, b.grad)  # 得到 dloss/dw 和 dloss/db
```

### 与 optimizer 的关系

手动写 `w -= lr * w.grad` 等价于一行：
```python
optimizer = torch.optim.SGD([w, b], lr=0.1)
optimizer.step()      # 执行更新
optimizer.zero_grad() # 清零梯度（替代手动 w.grad.zero_()）
```

> 🌰 **生活类比**：optimizer 就是"自动洗碗机"——以前要手动洗（`w -= lr * w.grad`），现在按下按钮就好。

<a id="train-template"></a>
### 5 步 PyTorch 训练标准流程（必背）

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        # ① 前向传播
        y_pred = model(x)
        # ② 计算损失
        loss = criterion(y_pred, y)
        # ③ 梯度清零
        optimizer.zero_grad()
        # ④ 反向传播
        loss.backward()
        # ⑤ 更新权重
        optimizer.step()
```

> 🌰 **学车口诀**：看路 → 发现偏 → 清空记忆 → 算方向 → 打方向。

### 关闭梯度的两个时机

| 场景 | 写法 |
|------|------|
| 模型推理（不需要梯度） | `with torch.no_grad():` |
| 微调时只更新部分层 | `for p in bert.parameters(): p.requires_grad = False` |
| 单个张量 detach | `tensor.detach()`（断掉计算图） |

### Autograd 工作原理（一图记忆）

```
 你写的代码                        PyTorch 偷偷干的事
─────────────                    ──────────────────────
 w = tensor(2.0, requires_grad=True)   ★ 标记为叶子节点
       │
 x = tensor(3.0)                       （不需要梯度）
       │
 y = w * x          ──────►       记录: y.grad_fn = MulBackward
       │                          构建计算图: y ← (×) ← w, x
 z = y + 1          ──────►       记录: z.grad_fn = AddBackward
       │                          z ← (+) ← y, 1
 loss = z ** 2      ──────►       loss.grad_fn = PowBackward

 loss.backward()    ──────►       沿图反向走一遍：
                                  d(loss)/d(z) = 2z
                                  d(loss)/d(y) = 2z · 1
                                  d(loss)/d(w) = 2z · 1 · x  → 写到 w.grad
```

> 🌰 **生活类比**：正向传播像"做菜全程录像"，反向传播像"按倒带追溯每一步加了多少盐"——录像就是计算图，倒带就是 `backward()`。

<a id="requires-grad"></a>
### `requires_grad` 的传染规则（重要）

| 输入 | 输出 `requires_grad` |
|------|---------------------|
| 至少一个输入 `requires_grad=True` | ✅ True（自动传染） |
| 所有输入都 `requires_grad=False` | ❌ False（不会被跟踪） |

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)                       # 默认 False
y = w * x
print(y.requires_grad)  # → True（被 w 传染）
```

> 🌰 **生活类比**：一锅汤里只要有一颗大蒜，整锅都有蒜味——只要有一个张量 `requires_grad=True`，下游全部都被跟踪。

### `.grad` 属性的 4 个常见状态

| 状态 | 含义 | 何时出现 |
|------|------|---------|
| `None` | 还没反向传播过 | 刚创建张量、`zero_grad()` 之后 |
| 有值 | 已经算过梯度 | `backward()` 之后 |
| 累加叠加 | 多次 `backward()` 没清零 | 忘了 `zero_grad()` 时 |
| 报错 "leaf tensor has no grad" | 试图取**非叶子节点**的 `.grad` | 中间张量需要 `retain_grad()` 才能查 |

```python
# 想看中间张量的梯度？
y = w * x
y.retain_grad()      # ⭐ 显式保留
loss = y ** 2
loss.backward()
print(y.grad)        # 现在能看了
```

### 梯度累加是 bug 还是 feature？

**bug 场景**：忘了 `zero_grad()` → 梯度爆炸、模型不收敛。

**feature 场景**（**梯度累积** Gradient Accumulation）：故意不清零，模拟更大 batch_size，**显存有限时常用**。

```python
accumulation_steps = 4   # 累积 4 步等效 batch×4

for i, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y) / accumulation_steps   # 损失要除 N
    loss.backward()                                       # 梯度累加
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()        # 每 4 步才更新
        optimizer.zero_grad()   # 然后清零
```

> 🌰 **生活类比**：搬家时一次只能搬 4 个箱子，但你想搬完一整车 16 个。**累 4 趟再算总账**，效果等同一次搬 16 个——梯度累积同理。

### `backward()` 默认会"烧掉"计算图

```python
loss = (w - 5) ** 2
loss.backward()          # ✅ OK
loss.backward()          # ❌ RuntimeError: 计算图已被释放！

# 解决方案 1：保留图
loss.backward(retain_graph=True)
loss.backward()          # ✅ 第二次也能跑

# 解决方案 2：每次重新前向
for _ in range(2):
    loss = (w - 5) ** 2  # 重新构建图
    loss.backward()
```

> 🌰 **生活类比**：火车票一般是"一次性票"，撕了就废（默认 `backward` 销毁图，省内存）。买月票（`retain_graph=True`）才能反复用，但更费钱（占内存）。

<a id="autograd-errors"></a>
### 5 个最常见报错速查

| 报错 | 原因 | 解决 |
|------|------|------|
| `element 0 of tensors does not require grad` | 对没标记 `requires_grad` 的张量调 `backward` | 创建时加 `requires_grad=True` |
| `grad can be implicitly created only for scalar outputs` | 对向量调 `backward()` | 改成 `loss.sum().backward()` |
| `Trying to backward through the graph a second time` | 计算图已释放 | 加 `retain_graph=True` 或重新前向 |
| `a leaf Variable that requires grad is being used in an in-place operation` | 对叶子张量做了原地修改（如 `w.add_(1)`） | 包到 `with torch.no_grad():` 里 |
| `RuntimeError: Found dtype Double but expected Float` | 张量类型不一致 | 统一用 `.float()` |

### Autograd 一句话记忆口诀

> **"叶子开梯度，正向画地图，标量调反向，梯度记得清，no_grad 推理，detach 断后路。"**


> 🌰 **生活类比**：`torch.no_grad()` = 打开计算器的"省电模式"，不再记录步骤，速度快 + 省内存。



<a id="activation-functions"></a>
## 四大激活函数

> 没有激活函数，神经网络再深也只是一连串线性变换，等价于一层。激活函数提供**非线性**，让网络能拟合复杂模式。下表对比 Sigmoid / Tanh / ReLU / Leaky ReLU 的特性和适用场景。

| 激活函数 | 公式 | 优点 | 缺点 | 适用场景 | 代码示例 |
|---------|------|------|------|----------|----------|
| **Sigmoid** | `σ(x) = 1/(1+e^(-x))` | 输出范围(0,1)，适合概率输出 | 梯度消失，输出非零中心 | 二分类输出层 | [01_激活函数sigmoid详解.py](Deep%20learnning/03-激活函数/01_激活函数sigmoid详解.py) |
| **Tanh** | `tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))` | 零中心化，收敛更快 | 梯度消失 | RNN隐藏层 | [02_激活函数tanh详解.py](Deep%20learnning/03-激活函数/02_激活函数tanh详解.py) |
| **ReLU** | `max(0, x)` | 计算简单，缓解梯度消失 | Dead ReLU问题（负区间梯度为0） | 大多数隐藏层（默认首选） | [03_激活函数relu详解.py](Deep%20learnning/03-激活函数/03_激活函数relu详解.py) |
| **Leaky ReLU** | `max(αx, x), α≈0.01` | 解决Dead ReLU | 效果不稳定，α需调参 | ReLU效果不佳时尝试 | — (ReLU 变体，见同文件) |

#### 激活函数选择建议

| 网络层次 | 推荐激活函数 | 原因 |
|---------|------------|------|
| **输入层** | 无需激活函数 | 直接传入原始特征 |
| **隐藏层** | ReLU / Leaky ReLU | 计算高效，缓解梯度消失 |
| **输出层-二分类** | Sigmoid | 输出概率值 [0,1] |
| **输出层-多分类** | Softmax | 输出概率分布 |
| **输出层-回归** | 无需激活函数 / Linear | 直接输出连续值 |
| **RNN/LSTM** | Tanh | 零中心化，稳定梯度 |

<a id="parameter-init"></a>
## 参数初始化（Parameter Initialization）— 给模型"一个合理的起点"

> 📂 文件来源：[Deep learnning/04-参数初始化/](Deep%20learnning/04-参数初始化/)

### 为什么参数初始化这么重要？

神经网络训练本质上是**找路**——从随机的起点出发，沿着梯度下降的方向一步步走到最优点。

| 初始化不好 | 初始化好 |
|-----------|---------|
| 🚫 梯度消失：深层梯度 ≈ 0，前面几层根本学不动 | ✅ 梯度通畅：信息在前向/反向传播中稳定流动 |
| 🚫 梯度爆炸：loss 变成 NaN，训练直接崩 | ✅ 数值稳定：loss 正常下降 |
| 🚫 收敛极慢：绕远路，花几倍时间才走到终点 | ✅ 收敛快：起点就在"好位置"附近 |

> 🌰 **生活类比**：登山团从山脚出发 → 体力消耗大、容易迷路（坏初始化）。直升机直接把你送到半山腰营地 → 轻松登顶（好初始化）。初始化就是那个直升机——省掉不必要的路程。

### PyTorch 7 种初始化方式速查表

| # | 方法 | API | 默认分布 | 一句话特点 | 适用场景 | 🔗 |
|---|------|-----|---------|-----------|---------|-----|
| 1 | **均匀初始化** | `init.uniform_(w)` | U(0,1) | 所有值等概率出现 | 简单基线，对称数据 | [代码](Deep%20learnning/04-参数初始化/05_参数初始化7种方式详解.py) |
| 2 | **正态初始化** | `init.normal_(w, mean, std)` | N(0,1) | 中心附近概率大，两边小 | 一般默认选项 | [代码](Deep%20learnning/04-参数初始化/05_参数初始化7种方式详解.py) |
| 3 | **全 0 初始化** | `init.zeros_(w)` | 全是 0 | 最简单但最坑 | ❌ 几乎所有场景都不推荐 | [代码](Deep%20learnning/04-参数初始化/05_参数初始化7种方式详解.py) |
| 4 | **全 1 初始化** | `init.ones_(w)` | 全是 1 | 对称破缺失败，等同 0 | ❌ 同全 0，基本不用 | [代码](Deep%20learnning/04-参数初始化/05_参数初始化7种方式详解.py) |
| 5 | **固定值初始化** | `init.constant_(w, val)` | 用户指定 | 你想设多少就设多少 | 特殊定制（如偏置设 0） | [代码](Deep%20learnning/04-参数初始化/05_参数初始化7种方式详解.py) |
| 6 | **Kaiming 初始化** | `init.kaiming_normal_(w)` | N(0, √(2/fan_in)) | ReLU 家族的黄金搭档 | **隐藏层用 ReLU 时首选** | [代码](Deep%20learnning/04-参数初始化/05_参数初始化7种方式详解.py) |
| 7 | **Xavier 初始化** | `init.xavier_normal_(w)` | N(0, √(1/fan_in)) | Sigmoid/Tanh 的标配 | 隐藏层用 Sigmoid/Tanh 时首选 | [代码](Deep%20learnning/04-参数初始化/05_参数初始化7种方式详解.py) |

> `fan_in` = 该层的输入神经元个数。Kaiming 除以 `√(2/fan_in)`，Xavier 除以 `√(1/fan_in)`，Kaiming 更激进一点——因为 ReLU 会杀死一半神经元。

---

### ① 均匀初始化 — 纯随机，听天由命

```python
linear = nn.Linear(5, 3)
nn.init.uniform_(linear.weight)   # 范围 [0, 1)
```

> 🌰 **生活类比**：从 0 到 1 的区间里闭着眼睛随便抓一个数，抓到多少算多少。

---

### ② 正态初始化 — 大多数取值在均值附近

```python
linear = nn.Linear(5, 3)
nn.init.normal_(linear.weight, mean=0, std=1)  # N(0, 1)
```

> 🌰 **生活类比**：全班身高分布——大部分人集中在平均身高附近，特别高和特别矮的很少。这种"中间多、两头少"的分布更自然。

---

### ③④ 全 0 / 全 1 初始化 — ❌ 新手最容易踩的坑

```python
linear = nn.Linear(5, 3)
nn.init.zeros_(linear.weight)   # 全部设成 0
nn.init.ones_(linear.weight)    # 全部设成 1
```

**为什么全 0 / 全 1 不行？**

所有神经元输出完全一样 → 反向传播梯度也一样 → 所有神经元学到同样的特征 → **模型退化成一个神经元**。

> 🌰 **生活类比**：全班同学都用同一份答案考试 → 每个人都考一样的分数 → 你分不清谁数学好谁语文好（参数没有"差异化"）。

**唯一的例外**：偏置 `bias` 通常初始化为 0，这是安全的。

---

### ⑤ 固定值初始化 — 自定义专属起点

```python
linear = nn.Linear(5, 3)
nn.init.constant_(linear.weight, 2.6)  # 所有权重 = 2.6
```

> 🌰 **生活类比**：公司给每个新员工发同样的起步工资 2.6 万——统一但缺乏个性。大多数场景下用随机初始化更合理。

---

### ⑥ Kaiming 初始化（He Initialization）— 🌟 最常用，ReLU 绝配

**发明背景**：2015 年何恺明发现，用 Xavier 初始化搭配 ReLU 效果不好，因为 ReLU 会把一半输出变 0，方差直接砍半。Kaiming 初始化专门补偿这个损失。

```python
linear = nn.Linear(5, 3)

# 正态版（推荐）
nn.init.kaiming_normal_(linear.weight)
# → N(0, √(2/fan_in))，fan_in = 输入维度

# 均匀版
nn.init.kaiming_uniform_(linear.weight)
# → U(-√(6/fan_in), +√(6/fan_in))
```

**数学直觉**：

| 初始化 | 方差 | 为什么 |
|--------|------|--------|
| Xavier | 1/fan_in | 假设激活函数是线性的（Sigmoid/Tanh 近似线性区） |
| **Kaiming** | **2/fan_in** | 假设用了 ReLU（一半输出归 0，方差减半，所以补一倍） |

> 🌰 **生活类比**：Kaiming 是"高海拔特供氧气瓶"——你知道爬珠峰（ReLU）氧气会稀薄一半，出发前就多背一倍。Xavier 是"普通登山氧气瓶"——走平缓坡（Sigmoid/Tanh）够用，上高峰就不行了。

---

### ⑦ Xavier 初始化（Glorot Initialization）— Sigmoid/Tanh 时代的经典

**发明背景**：2010 年 Glorot 发现，简单正态初始化在深层网络中梯度会消失，他推导出方差应为 `1/fan_in` 才能让信息在层间稳定流动。

```python
linear = nn.Linear(5, 3)

# 正态版
nn.init.xavier_normal_(linear.weight)
# → N(0, √(1/fan_in))

# 均匀版
nn.init.xavier_uniform_(linear.weight)
# → U(-√(3/fan_in), +√(3/fan_in))
```

> 🌰 **生活类比**：Xavier 像是"自适应水管"——水龙头水流太猛会冲坏花（梯度爆炸），水流太细浇不到远端（梯度消失），它把水管粗细调到刚刚好，让水流均匀地浇灌每一层。

---

### 一张图记住怎么选

```
你的激活函数是什么？
├─ ReLU / Leaky ReLU / PReLU ──► Kaiming 初始化（当前最常用）
├─ Sigmoid / Tanh ─────────────► Xavier 初始化
├─ 其他 / 不确定 ──────────────► 正态初始化 N(0, 0.01)
│                                  （保险选项，数值小防止爆炸）
└─ 偏置 bias ──────────────────► constant_(bias, 0)（几乎总是 0）
```

**实战口诀**：
> **"ReLU 配 Kaiming，Sigmoid 配 Xavier，偏置全 0，权重不取整（别用 constant）"**

### 实际项目中怎么做？

PyTorch 的 `nn.Linear` / `nn.Conv2d` 等层**自带默认初始化**（Kaiming Uniform），大多数时候你**不需要手动调**。手动初始化在以下场景才需要：

1. **自定义层**：自己写了 `nn.Module`，没有内置初始化逻辑
2. **迁移/微调**：想用特定分布重新初始化某些层
3. **调试研究**：验证初始化对收敛的影响

```python
# 实际项目中的标准写法（只需要覆盖默认时用）
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
model.apply(init_weights)  # 一键应用到所有子层
```

### 面试高频题

1. **Q：为什么不能全 0 初始化？**
   A：对称性问题——所有神经元输出相同、梯度相同、学到的特征相同，模型退化成一个神经元。

2. **Q：Kaiming 和 Xavier 的核心区别？**
   A：方差公式不同。Kaiming 方差 = 2/fan_in（补偿 ReLU 砍半的方差），Xavier 方差 = 1/fan_in（假设线性激活）。

3. **Q：偏置 bias 一般初始化为多少？**
   A：几乎 always 0。偏置的作用是平移，初始为 0 让网络自己学偏移量。

4. **Q：`init.kaiming_normal_` 后面的下划线什么意思？**
   A：PyTorch 约定**带下划线 = 原地修改（in-place）**。函数直接修改传入张量的值，不返回新对象。

5. **Q：现实中我真的需要手动初始化吗？**
   A：**大多数场景不需要**。PyTorch 内置层已经自带合理的默认初始化（Linear → Kaiming Uniform，LSTM → Uniform 等）。只有自定义层或特殊需求时才手动调。


<a id="model-optimization"></a>
## 模型优化（Model Optimization）— 训练快、收敛稳、防过拟合

> 📂 文件来源：[Deep learnning/05_模型优化/](Deep%20learnning/05_模型优化/)

模型优化是深度学习的"调优三板斧"：**选对优化器 → 调好学习率 → 加正则化**。本章从指数加权平均（EWMA）这个底层思想出发，依次搞懂优化器演进、学习率策略、Dropout 和 BatchNorm。

---

### 一、指数加权平均（EWMA）— 所有高级优化器的地基

**核心公式**：

```
v_t = β · v_{t-1} + (1 - β) · θ_t
```

- `v_t`：当前时刻的加权平均值
- `θ_t`：当前时刻的原始数据
- `β`：历史权重（0~1），**β 越大曲线越平缓**

**β 的影响**：

| β 值 | 含义 | 效果 |
|------|------|------|
| 0.9  | ≈ 平均最近 10 个数据 | 曲线平滑，保留大致趋势 |
| 0.99 | ≈ 平均最近 100 个数据 | 曲线非常平滑，但反应滞后 |
| 0.5  | ≈ 平均最近 2 个数据 | 几乎跟随原始数据，不平滑 |

> 🌰 **生活类比**：你记录每天的体重。β=0.9 相当于"昨天的估计值占 90% + 今天实测占 10%"，数字有波动但趋势看得清。β=0.99 几乎只看历史趋势，你今天胖了 2 斤要好多天才能反映出来。

> <small>**EWMA 在优化器中的作用**：梯度下降时，用 EWMA 对当前梯度做平滑（`v = β·v + (1-β)·∇w`），相当于给梯度加了"惯性"。方向相反的震荡步被抵消，方向一致的下坡步加速通过，整体更新更平稳、收敛更快。动量法、RMSprop、Adam 都依赖这个思想。</small>

**为什么 EWMA 是优化器的地基？**

动量法（Momentum）用它平滑梯度，RMSprop 用它平滑梯度平方，Adam 两个都用。**掌握了 EWMA，就掌握了优化器演进的钥匙。**

```python
# 手动实现 EWMA（核心 3 行）
beta = 0.9
v = 0
for t in data:
    v = beta * v + (1 - beta) * t   # ← 指数加权平均
```

---

### 二、梯度下降优化器演进 — 从 SGD 到 Adam 的升级之路

> 📂 [02_回顾梯度下降_SGD.py](Deep%20learnning/05_模型优化/02_回顾梯度下降_SGD.py) + [03_梯度下降优化器_动量法_adagrad_RMSprop_Adam.py](Deep%20learnning/05_模型优化/03_梯度下降优化器_动量法_adagrad_RMSprop_Adam.py)

每一代优化器都是为了**解决上一代的核心痛点**：

```
SGD（最朴素）
  │   ❌ 更新方向震荡、收敛慢
  ├─→ Momentum（动量法）
  │      引入 EWMA 平滑梯度方向，减少震荡
  │   ❌ 所有参数共用同一学习率
  ├─→ Adagrad（自适应学习率）
  │      每个参数独立调整学习率，稀疏特征更新大，频繁特征更新小
  │   ❌ 学习率过早衰减到接近 0
  ├─→ RMSprop（改进 Adagrad）
  │      用 EWMA 替换 Adagrad 的累加求和，缓解学习率衰减问题
  │   ❌ 缺少动量
  └─→ Adam（集大成者）
          Momentum + RMSprop = 当前首选
```

#### ① SGD — 最朴素的梯度下降

```python
optimizer = torch.optim.SGD([w], lr=0.01)
# 更新规则: w = w - lr * ∇w
```

| 优点 | 缺点 |
|------|------|
| 简单，容易理解 | 梯度方向震荡（尤其在峡谷地形） |
| 内存占用少 | 所有参数共享同一学习率 |
| 理论成熟 | 在平坦区域收敛极慢 |

> 🌰 **生活类比**：SGD 像蒙着眼睛下山——每走一步都选最陡的方向，但可能会在谷底来回震荡走不出去。

#### ② Momentum（动量法）— 引入惯性

```python
optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)  # SGD + 动量
```

**核心思想**：用 EWMA 平滑梯度方向，保留历史梯度的一部分。

```
v_t     = β · v_{t-1} + (1-β) · ∇w_t    ← 梯度的指数加权平均
w_{t+1} = w_t - lr · v_t                  ← 用平滑后的梯度更新
```

| 优点 | 缺点 |
|------|------|
| 减少震荡，收敛更平稳 | 多了一个超参 β（默认 0.9） |
| 能"冲过"局部极小点和平坦区 | 在梯度变化剧烈的场景可能 overshoot |

> 🌰 **生活类比**：下坡时推一个铁球——铁球有惯性（动量），即便路有起伏也能冲过去，不会像羽毛（SGD）一样随风乱飘。

#### ③ Adagrad — 让每个参数有自己的学习率

```python
optimizer = torch.optim.Adagrad([w], lr=0.01)
```

**核心思想**：频繁更新的参数学习率变小，稀疏更新的参数学习率变大。

```
G_t     = G_{t-1} + (∇w_t)²              ← 梯度平方的累加
w_{t+1} = w_t - lr / (√G_t + ε) · ∇w_t    ← 学习率被 G_t 缩放
```

| 优点 | 缺点 |
|------|------|
| 自适应学习率，适合稀疏特征 | **学习率过早衰减**——G_t 不断增大，学习率趋向 0 |
| 无需手动调整学习率 | 训练后期基本停止学习 |

> 🌰 **生活类比**：Adagrad 像一个"越学越慢"的学生——每学到一个新知识就记一笔，笔记越积越厚（G_t 累加），翻书越来越慢。适合突击复习（短时间训练），不适合长期学习。

#### ④ RMSprop — 修复 Adagrad 的"过早衰减"

```python
optimizer = torch.optim.RMSprop([w], lr=0.01, alpha=0.99)
```

**核心思想**：把 Adagrad 的累加求和换成 EWMA，不让分母无限增长。

```
G_t     = β · G_{t-1} + (1-β) · (∇w_t)²  ← EWMA 替代累加
w_{t+1} = w_t - lr / (√G_t + ε) · ∇w_t
```

| 优点 | 缺点 |
|------|------|
| 解决 Adagrad 过早衰减问题 | 多了一个超参 α（默认 0.99） |
| 自适应学习率，训练稳定 | RMSprop 单独用效果不如 Adam |

> 🌰 **生活类比**：RMSprop 像一个"复习有方"的学生——笔记只保留最近的笔记（EWMA），不会越积越厚，能持续学习新知识。

#### ⑤ Adam（Adaptive Moment Estimation）— 🌟 当前默认首选

```python
optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.99))
```

**核心思想**：Adam = **动量法（一阶矩）+ RMSprop（二阶矩）**，两个 EWMA 分工合作。

```
一阶矩（动量）: m_t = β₁·m_{t-1} + (1-β₁)·∇w_t    ← 平滑梯度方向
二阶矩（RMS） : v_t = β₂·v_{t-1} + (1-β₂)·(∇w_t)² ← 自适应学习率

w_{t+1} = w_t - lr · m_t / (√v_t + ε)
```

| 优点 | 缺点 |
|------|------|
| ✅ 自适应学习率 + 动量，**绝大多数场景效果最好** | 需要更多内存（存一阶/二阶矩） |
| ✅ 超参鲁棒（即使不太调参也能收敛） | 在极端稀疏场景不如 Adagrad |
| ✅ 默认超参 `lr=0.001, betas=(0.9, 0.999)` 适配大部分任务 | 相比 SGD 泛化性略有争议（但实战差异很小） |

> 🌰 **生活类比**：Adam 像一个"聪明又稳重的司机"——动量法告诉你要往哪个方向走（惯性），RMSprop 告诉你要走多快（路况自适应）。两者配合，又快又稳。

#### 优化器对比总结

| 优化器 | 自适应 LR | 动量 | 核心超参 | 适用场景 |
|--------|----------|------|---------|---------|
| **SGD** | ❌ | ❌ | lr | 简单任务、小网络、追求泛化性 |
| **Momentum** | ❌ | ✅ | lr, momentum(0.9) | 大部分基础任务 |
| **Adagrad** | ✅ | ❌ | lr | 稀疏数据、NLP 词嵌入 |
| **RMSprop** | ✅ | ❌ | lr, alpha(0.99) | 序列模型、RL 强化学习 |
| **Adam** | ✅ | ✅ | lr, betas=(0.9,0.999) | **默认首选，新手直接无脑用** |

> **实战建议**：没有特别理由，**直接选 Adam**。它像智能手机——什么都能干、默认设置就很好用。想极致调优时再考虑其他。

---

### 三、学习率 — 最敏感的超参

> 📂 [04_学习率衰减三大方式对比.py](Deep%20learnning/05_模型优化/04_学习率衰减三大方式对比.py) + [05_拓展_通过调整学习率展示各种问题对应图表.py](Deep%20learnning/05_模型优化/05_拓展_通过调整学习率展示各种问题对应图表.py)

#### 不同学习率的效果

| 学习率 | 现象 | 比喻 |
|-------|------|------|
| ≤ 0.01 | 下降太慢，收敛时间长 | 乌龟跑步 |
| **0.05~0.1** | **正常下降，稳定收敛** | 正常步行 |
| 0.125 | 一步到位（适合简单函数） | 跳远刚好踩线 |
| 0.2 | 在最优值附近震荡，下不去 | 钟摆来回晃 |
| ≥ 0.3 | **梯度爆炸，loss 变成 NaN** | 火箭冲出大气层 |

> 💡 **核心结论**：学习率太小 ≈ 没在学，学习率太大 ≈ 学崩了。**合理区间通常是 0.001~0.1**，Adam 默认 0.001 就很好用。

#### 学习率衰减（Learning Rate Scheduler）

训练后期需要让学习率逐渐减小，避免在最优点附近震荡。PyTorch 提供 3 种主流方式：

| 方式 | API | 行为 | 适用场景 |
|------|-----|------|---------|
| **等间隔衰减** | `StepLR(optimizer, step_size, gamma)` | 每 `step_size` 轮 × gamma | 固定节奏训练 |
| **指定间隔衰减** | `MultiStepLR(optimizer, milestones, gamma)` | 在指定轮次 × gamma | 自定义衰减曲线 |
| **指数衰减** | `ExponentialLR(optimizer, gamma)` | 每轮 × gamma，平滑衰减 | 需要平滑衰减时 |

> 🌰 **生活类比**：
> - **等间隔衰减** = 每月工资打 9 折（固定节奏）
> - **指定间隔衰减** = 入职 25 天、125 天、175 天时分别调薪（自定义节点）
> - **指数衰减** = 工资每天降一点（平滑过渡）

```python
# 完整模板：优化器 + 学习率衰减
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(200):
    train_one_epoch(model, dataloader, optimizer)  # 常规训练
    scheduler.step()                                # 更新学习率（每 epoch 后）
```

---

### 四、Dropout 随机失活 — 最简单的防过拟合

> 📂 [06_dropout随机失活正则化.py](Deep%20learnning/05_模型优化/06_dropout随机失活正则化.py)

**核心思想**：训练时**随机丢弃**一部分神经元，强迫网络不依赖单个特征。

```python
linear = nn.Linear(4, 5)
dropout = nn.Dropout(p=0.5)   # 50% 概率丢弃

x = torch.randn(1, 4)
x = linear(x)                  # 加权求和
x = torch.relu(x)              # ReLU 激活
x = dropout(x)                 # ★ 随机丢弃一半神经元（输出置 0）
```

**关键理解**：

| 阶段 | Dropout 行为 | 原理 |
|------|-------------|------|
| **训练时** | 以概率 p 随机丢弃神经元（输出置 0） | 防止共适应（co-adaptation） |
| **推理时** | Dropout 关闭，所有权重 × (1-p) 缩放 | 保持期望输出一致 |

> 🌰 **生活类比**：Dropout 像"团队轮岗"——每次训练随机抽掉一半员工，剩下的必须独立完成工作。这样每个人都能独当一面，不会过度依赖特定同事（防止过拟合）。

**使用建议**：
- `p=0.5` 是常见默认值（对隐藏层）
- **Dropout 放激活函数之后**
- 推理时用 `model.eval()` 自动关闭 Dropout
- 过拟合明显时才加，欠拟合时不要加

---

### 五、批量归一化（Batch Normalization）— 训练加速器

> 📂 [07_批量归一化.py](Deep%20learnning/05_模型优化/07_批量归一化.py)

**核心问题**：深层网络中，每层输入的分布不断变化（内部协变量偏移），导致：
- 上层参数要不断适应下层输出的分布变化
- 需要更小的学习率才能稳定训练
- 梯度消失问题加剧

**BatchNorm 解决方案**：把每层的输入拉回标准正态分布 N(0,1)，再学两个可恢复参数 γ（缩放）和 β（偏移）。

```
x̂ = (x - μ) / √(σ² + ε)      ← 归一化到 N(0,1)
y = γ · x̂ + β                 ← 可学习的恢复（想恢复多少学多少）
```

| 参数 | 含义 | 是否可学习 |
|------|------|-----------|
| μ, σ | 当前 batch 的均值和标准差 | ❌ 统计得到 |
| γ | 缩放因子（学回来的标准差） | ✅ 可学习 |
| β | 偏移量（学回来的均值） | ✅ 可学习 |
| ε | 防止除 0 的小常数（默认 1e-5） | ❌ 固定值 |

> 🌰 **生活类比**：BatchNorm 像"标准化考试"——把所有学生的分数调到均分 70 分（归一化），再允许部分学霸恢复高分（γ 放大），部分学渣保持低分（β 偏移）。**先统一标准，再恢复差异**。

```python
# BatchNorm2d: 用于 CNN（输入 NCHW）
bn = nn.BatchNorm2d(2)         # 2 = 通道数
input = torch.randn(1, 2, 3, 4)  # (N, C, H, W)
output = bn(input)

print(bn.weight)  # γ，初始全 1
print(bn.bias)    # β，初始全 0
```

**训练 vs 推理的差异**：

| 阶段 | μ 和 σ 怎么来 | 含义 |
|------|--------------|------|
| **训练时** | 当前 batch 的统计值 | 让 BN 适配当前数据分布 |
| **推理时** | 训练阶段累积的**滑动平均** | 使用全局稳定的统计数据 |

> ⚠️ **必须调用 `model.eval()` 切换推理模式**，否则 BN 层的 μ/σ 在推理时仍用 batch 统计，结果会不稳定。

**使用建议**：
- 放在激活函数**之前**（原始论文做法）或**之后**（某些现代做法），通常之前
- CNN 用 `BatchNorm2d`，全连接用 `BatchNorm1d`
- BatchNorm 自带正则化效果，有时可替代 Dropout

---

### 六、模型优化总览图

```
                   指数加权平均 (EWMA)
                          │
            ┌─────────────┴─────────────┐
            ▼                            ▼
      ┌─ 优化器 ──┐               ┌─ 学习率 ──┐
      │ SGD       │               │ 大小选择   │
      │ Momentum  │               │ StepLR    │
      │ Adagrad   │               │ MultiStep │
      │ RMSprop   │               │ ExpLR     │
      │ Adam 🌟   │               │           │
      └───────────┘               └───────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                            ▼
    ┌─ 正则化 ──┐              ┌─ 归一化 ──┐
    │ Dropout   │              │ BatchNorm │
    │ 随机失活   │              │ 层归一化   │
    └───────────┘              └───────────┘
```

### 面试高频题

1. **Q：Adam 相比 SGD 好在哪里？为什么现在都默认用 Adam？**
   A：Adam = Momentum（平滑梯度方向）+ RMSprop（自适应学习率），对超参不敏感，大部分场景直接收敛，无需精细调参。

2. **Q：学习率太大 / 太小会怎样？**
   A：太小 → 收敛极慢甚至停滞；太大 → 震荡不收敛甚至梯度爆炸（loss 变 NaN）。

3. **Q：Dropout 训练和推理有什么区别？**
   A：训练时随机丢弃；推理时不丢弃但权重 × (1-p) 保持期望一致。`model.eval()` 自动处理。

4. **Q：为什么需要学习率衰减？**
   A：训练初期需要大学习率快速下降，训练后期需要小学习率精细收敛。不衰减会在最优点附近震荡。

5. **Q：BatchNorm 训练和推理有什么区别？**
   A：训练时 μ/σ 用当前 batch 统计；推理时用全局滑动平均。必须切换 `model.eval()` / `model.train()` 模式。

6. **Q：BN 层中的 γ 和 β 是什么？**
   A：γ（缩放）和 β（偏移）是可学习参数，用于"恢复"归一化后丢失的表达能力。初始化 γ=1, β=0。


# NLP

> 自然语言处理（Natural Language Processing）让计算机"看懂"和"会说"人话。本章按经典学习顺序：**预处理 → 词表示（One-Hot → Word2Vec [CBOW/Skip-gram] → Embedding）→ RNN 家族 → 注意力机制 → Transformer**，每一步都解决前一步的痛点。

<a id="text-preprocessing"></a>
## 文本预处理流程

| 步骤 | 方法 | 工具/库 | 说明 |
|------|------|---------|------|
| **1. 分词** | jieba分词、空格分割 | `jieba`、`split()` | 将句子切分为词语 |
| **2. 去停用词** | 过滤常见无意义词 | 停用词表 | 去除“的”、“是”等 |
| **3. 词性标注** | POS Tagging | `jieba.posseg` | 标注名词、动词等 |
| **4. 命名实体识别** | NER | spaCy、NLTK | 识别人名、地名、组织名 |
| **5. 向量化** | One-Hot、Word2Vec、Embedding | gensim、torch.nn.Embedding | 将词转为向量 |

## 词表示方法对比

| 方法 | 维度 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **One-Hot** | 词汇表大小 | 简单直观 | 稀疏、无语义信息 | 小规模数据集 |
| **TF-IDF** | 词汇表大小 | 考虑词频和文档频率 | 仍为稀疏向量 | 文本分类、检索 |
| **Word2Vec** | 自定义(50-300) | 捕捉语义关系 | 静态嵌入，一词一义 | 词相似度、类比任务 |
| **GloVe** | 自定义(50-300) | 全局统计信息 | 静态嵌入 | 通用词嵌入 |
| **BERT Embedding** | 768/1024 | 上下文相关，动态 | 计算成本高 | 现代NLP任务 |

<a id="word2vec-cbow-skipgram"></a>
## Word2Vec 深入：CBOW 与 Skip-gram

> Word2Vec (Mikolov et al., 2013) 是 NLP 历史上最具影响力的词向量模型。它用**浅层神经网络**把词映射到低维稠密向量，让语义相近的词在向量空间中距离也近。本节深入拆解它的两套训练架构——CBOW 和 Skip-gram。

### 一、核心思想（一张图读懂两种架构）

```
         CBOW                                     Skip-gram
    "用周围词 → 猜中间词"                      "用中间词 → 猜周围词"

   输入: 前后各 2 个词                          输入: 1 个中心词
   输出: 中间那个词                             输出: 周围多个词

   ┌───────────────────────┐                ┌───────────────────────┐
   │  w(t-2) w(t-1) w(t+1) w(t+2)  │        │            w(t)            │
   │   ↓      ↓      ↓      ↓     │        │        ↓  ↓  ↓  ↓       │
   │   └──────┴──────┴──────┘     │        │   w(t-2) w(t-1) w(t+1) w(t+2) │
   │          ↓ 求和/平均          │        │   上下文词 = 被预测的目标        │
   │        ┌──────┐              │        └───────────────────────┘
   │        │  w(t) │   ← 预测目标  │
   │        └──────┘              │
   └───────────────────────┘

    🌰 生活类比:                              🌰 生活类比:
    "你前后的朋友都是什么人，                   "你是什么人，就招什么样的朋友"
     你就是什么人"
```

> 💡 **一句话区别**：CBOW 是"**众人猜一人**"——周围词投票决定中心词；Skip-gram 是"**一人猜众人**"——中心词发散预测周围词。方向相反，用的网络结构和损失函数也不一样。

### 二、CBOW（Continuous Bag of Words）— 用上下文预测中心词

#### 2.1 训练过程（4 步）

```
输入句子: "我 爱 吃 四川 火锅"
窗口大小 = 2（中心词左右各取 2 个词）

Step 1: 取中心词 "吃"，上下文 = ["我", "爱", "四川", "火锅"]
        上下文词       中心词
        ["我","爱","四川","火锅"] → "吃"

Step 2: 查 Embedding 表，把 4 个上下文词各自转成向量
        v_我, v_爱, v_四川, v_火锅  各是 d 维向量

Step 3: 对这 4 个向量求平均 → 得到 1 个上下文向量 h
        h = (v_我 + v_爱 + v_四川 + v_火锅) / 4

Step 4: 用 h 做 Softmax 分类，预测中心词是哪个
        损失 = CrossEntropy(预测概率, 真实词"吃")
        反向传播 → 更新 Embedding 表
```

> ⚡ **CBOW 的关键在于"平均"**：上下文词的顺序被忽略（bag-of-words），所有上下文词被一视同仁地求和平均。这就是名字里的 "Bag of Words" 来源。

#### 2.2 代码骨架（PyTorch 手工实现 CBOW）

```python
import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)   # 输入词向量矩阵
        self.linear = nn.Linear(embed_dim, vocab_size)          # 输出投影层

    def forward(self, context_words):
        # context_words: [batch, context_size] 上下文词的索引
        embeds = self.embeddings(context_words)    # [batch, ctx, dim]
        h = embeds.mean(dim=1)                     # ★ 关键：平均所有上下文词向量
        out = self.linear(h)                       # [batch, vocab_size] 预测中心词
        return out

# 使用示例
vocab_size, embed_dim = 10000, 100
model = CBOW(vocab_size, embed_dim)

# 输入：周围 4 个词的索引
context = torch.tensor([[12, 45, 789, 54], [3, 67, 234, 89]])  # [2, 4]
target  = torch.tensor([128, 456])                                # [2] 中心词索引

logits = model(context)                      # [2, 10000]
loss = nn.CrossEntropyLoss()(logits, target)
loss.backward()
```

### 三、Skip-gram — 用中心词预测上下文

#### 3.1 训练过程（4 步）

```
输入句子: "我 爱 吃 四川 火锅"
窗口大小 = 2

Step 1: 取中心词 "吃"，上下文目标 = ["我", "爱", "四川", "火锅"]
        中心词           上下文词
        "吃"  →  ["我", "爱", "四川", "火锅"]

Step 2: 查 Embedding 表，把中心词转成向量
        v_吃  是 d 维向量

Step 3: 用 v_吃 预测每一个上下文词（4 次独立的 Softmax 分类）
        P(我|吃), P(爱|吃), P(四川|吃), P(火锅|吃)

Step 4: 损失 = 4 个预测的 CrossEntropy 之和
        反向传播 → 更新 Embedding 表
```

> ⚡ **Skip-gram 的关键在于"一对多"**：一个中心词产生多个训练样本，每个训练样本是 `(中心词, 上下文词)` 对。相比 CBOW，Skip-gram 对每个中心词的利用更充分——每个上下文关系都单独学习。

#### 3.2 代码骨架（PyTorch 手工实现 Skip-gram）

```python
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embeddings  = nn.Embedding(vocab_size, embed_dim)  # 中心词向量
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)  # 上下文词向量

    def forward(self, center_word):
        # center_word: [batch] 中心词索引
        embeds = self.in_embeddings(center_word)   # [batch, dim]
        out = embeds @ self.out_embeddings.weight.T # [batch, vocab_size]
        return out

    def get_word_vectors(self):
        # 最终词向量通常取 in_embeddings，或两个 Embedding 的平均
        return self.in_embeddings.weight.data

# 使用示例
model = SkipGram(vocab_size, embed_dim)

# 输入：中心词索引
center = torch.tensor([128, 456])                 # [2]
# 标签：上下文词索引（每个中心词对应一个上下文词）
context_target = torch.tensor([12, 67])            # [2]

logits = model(center)                            # [2, 10000]
loss = nn.CrossEntropyLoss()(logits, context_target)
loss.backward()
```

> 💡 **两个 Embedding 矩阵**：Skip-gram 的标准实现有**输入矩阵**（存中心词向量）和**输出矩阵**（存上下文词向量），最终词向量通常取输入矩阵，或取两者的平均。

### 四、CBOW vs Skip-gram 对比总结

| 对比维度 | CBOW | Skip-gram |
|---------|------|-----------|
| **任务方向** | 上下文 → 中心词 | 中心词 → 上下文 |
| **训练速度** | ⚡ **快**（一次预测 1 个词） | 🐢 慢（一次预测 2k 个词，k=窗口大小） |
| **对低频词的效果** | 一般（低频词容易被"平均掉"） | ✅ **好**（每个上下文对都单独训练） |
| **对高频词的倾向** | 更强（频繁共现的词权重大） | 相对平衡 |
| **适合的数据量** | 大数据集（追求速度） | 小数据集（追求质量） |
| **向量质量** | 略低 | ✅ **更优**（尤其低频词向量） |
| **训练样本数** | ≈ 语料中的词数 | ≈ 语料词数 × (2 × 窗口大小) |

#### 选择建议

```
你的场景是什么？
├─ 数据量巨大（GB 级）+ 追求训练速度  ──► CBOW
├─ 数据量较小（MB 级）+ 追求词向量质量 ──► Skip-gram
├─ 有很多低频词（专业术语、罕见词）    ──► Skip-gram
└─ 不确定                             ──► Skip-gram（默认推荐）
```

> 🌰 **生活类比**：
> - **CBOW** = 阅卷老师看全班平均分来猜某个学生的成绩——快，但模糊。
> - **Skip-gram** = 班主任一个个找学生谈话了解情况——慢，但精准，连"偏科"（低频词特征）都能摸清楚。

### 五、两大训练加速技术（面试重点）

Word2Vec 如果直接做全词表 Softmax（V 类分类），每次前向和反向的复杂度是 O(V)，词汇表 V 通常几十万到几百万——**根本算不动**。以下两种技术把复杂度降到 O(log V) 或 O(k)。

#### 5.1 层次 Softmax（Hierarchical Softmax）

```
核心思想: 把 V 分类 → 变成"二叉树猜路径"（log₂V 次二分类）

              根节点
             /      \
           /          \
        [路径编码]   [路径编码]
       /    \         /    \
     词1   词2      词3   词4
    (001) (010)   (011) (100)

原来: 一次从 V 个候选里选出 1 个   → 复杂度 O(V)
现在: 沿着树走 log₂V 步，每步选左/右 → 复杂度 O(log V)
```

| 构造方式 | 说明 |
|---------|------|
| **哈夫曼树（Huffman Tree）** | Word2Vec 默认方式，高频词路径短（离根近），低频词路径长。高频词算得快，整体效率更高 |

> 🌰 **生活类比**：全词表 Softmax 像"V 选 1"的单选题，层次 Softmax 像做 log₂V 道"左还是右"的判断题——题多了一道，但每道只做对/错判断，快很多。

#### 5.2 负采样（Negative Sampling）

```
核心思想: 不跟所有 V 个词比，只跟"1 个正样本 + k 个负样本"比

每一轮训练:
  ┌─ 正样本（必须算）：真实的上下文词         → 1 个
  └─ 负样本（采样子集）：从语料中随机抽的不相关词 → k 个（通常 5~20）

损失变成: 对正样本说"相关性高"，对负样本说"相关性低"
复杂度: O(V) → O(k)，k 通常 5~20
```

**负样本的采样不是均匀随机**——Word2Vec 用**频率的 3/4 次方**作为采样概率：

```
P(词) = count(词)^(3/4) / Σ count(词ᵢ)^(3/4)
```

| 采样方式 | 效果 |
|---------|------|
| 均匀采样 | 高频词容易被抽为负样本，但高频词通常是停用词（"的"、"是"），信息量低 |
| **3/4 次方采样** | 适当提升低频词被抽中的概率，让训练更有信息量 |

> 🌰 **生活类比**：
> - **全词表 Softmax** = 相亲时把全国所有人都看一遍再决定跟谁
> - **负采样** = 只跟相亲对象（正样本）+ 随机问 5 个路人"这人靠谱吗"（负样本），效率飙升

#### 5.3 两种加速方式的选择

| 技术 | 复杂度 | 适用场景 | 优点 |
|------|--------|---------|------|
| **层次 Softmax** | O(log V) | 低频词较多的语料 | 精确，不需手动设 k |
| **负采样** | O(k) | 高频词较多的语料 | 实现简单，效果通常更好 |
| **现代默认** | — | **负采样**（几乎成为标准） |

### 六、Gensim 一行训练（实战速查）

```python
from gensim.models import Word2Vec

# 1. 准备语料：每行为一个分好词的句子列表
sentences = [['我', '爱', '吃', '四川', '火锅'],
             ['机器', '学习', '很', '有趣'],
             ['深度', '学习', '改变', '世界']]

# 2. 训练 CBOW
model_cbow = Word2Vec(sentences, vector_size=100, window=5, sg=0,  # sg=0 → CBOW
                      min_count=1, workers=4, epochs=10)

# 3. 训练 Skip-gram
model_sg = Word2Vec(sentences, vector_size=100, window=5, sg=1,    # sg=1 → Skip-gram
                    min_count=1, workers=4, epochs=10,
                    hs=0, negative=5, ns_exponent=0.75)             # 负采样参数

# 4. 查词向量 & 相似度
vec_火锅 = model_sg.wv['火锅']                            # 获取词向量
model_sg.wv.most_similar('火锅', topn=5)                  # 语义最相近的 5 个词
model_sg.wv.most_similar(positive=['国王', '女'], negative=['男'])  # 经典类比: 国王-男+女 ≈ 女王

# 5. 保存 & 加载
model_sg.save('word2vec.model')
model_sg.wv.save_word2vec_format('vectors.bin', binary=True)  # 只存向量，兼容其他工具
```

### 七、面试高频题

1. **Q：CBOW 和 Skip-gram 的核心区别是什么？**
   A：方向相反。CBOW 用上下文预测中心词（快，对高频词好），Skip-gram 用中心词预测上下文（慢但质量高，对低频词友好）。

2. **Q：为什么 Skip-gram 对低频词效果更好？**
   A：CBOW 对上下文词取平均，低频词的特征容易被淹没；Skip-gram 每个中心词-上下文对都独立训练，低频词的每个出现都被充分学习。

3. **Q：为什么需要负采样？不采样直接用 Softmax 行不行？**
   A：词汇表 V 通常几十万+，全 Softmax 每次 O(V) 计算量根本跑不动。负采样每次只算 1 + k 个词的 Softmax，复杂度 O(k)，k 通常 5~20。

4. **Q：负采样中为什么用 3/4 次方而不是均匀采样？**
   A：均匀采样会让高频停用词（"的"、"是"）反复被抽为负样本，但这些词的语义信息量低。3/4 次方压缩了高频词的采样概率，让低频词有更多机会被抽中，训练更有效。

5. **Q：Word2Vec 得到的词向量是静态的还是动态的？**
   A：**静态**。一个词只有一个固定向量，不管它出现在什么上下文里。比如 "苹果" 在 "吃苹果" 和 "买苹果手机" 中是同一个向量——这也是后来 BERT 等动态嵌入要解决的问题。

<a id="rnn-family"></a>
## RNN 循环神经网络家族

<a id="rnn"></a>
### 1. RNN (Recurrent Neural Network)
**核心思想**：引入时间维度，隐藏状态在不同时间步之间传递，实现序列记忆。

**公式**：
- `h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)`
- `y_t = softmax(W_hy * h_t + b_y)`

**优点**：
- ✅ 能处理变长序列数据
- ✅ 参数共享，减少参数量
- ✅ 理论上可以捕捉任意长度的依赖关系

**缺点**：
- ❌ **梯度消失/爆炸**：难以学习长距离依赖（>10个时间步）
- ❌ 训练速度慢，无法并行计算
- ❌ 短期记忆问题

**适用场景**：短序列建模、简单时序预测

**代码示例**：
```python
import torch.nn as nn

# batch_first=False: (seq_len, batch, input_size)
# batch_first=True: (batch, seq_len, input_size)
rnn = nn.RNN(input_size=5, hidden_size=7, num_layers=1, batch_first=False)
output, hn = rnn(x)  # output: 所有时间步输出, hn: 最后隐藏状态
```

---

<a id="lstm"></a>
### 2. LSTM (Long Short-Term Memory)
**核心思想**：引入**门控机制**和**细胞状态（Cell State）**，解决长期依赖问题。

**三大门控**：
| 门控 | 作用 | 公式 |
|------|------|------|
| **遗忘门 (Forget Gate)** | 决定丢弃哪些旧信息 | `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)` |
| **输入门 (Input Gate)** | 决定存储哪些新信息 | `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)` |
| **输出门 (Output Gate)** | 决定输出哪些信息 | `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)` |

**细胞状态更新**：
- `C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)` （候选值）
- `C_t = f_t * C_{t-1} + i_t * C̃_t` （更新细胞状态）
- `h_t = o_t * tanh(C_t)` （输出隐藏状态）

**优点**：
- ✅ 有效缓解梯度消失，能捕捉长距离依赖（100+时间步）
- ✅ 门控机制让模型学会“记住什么”和“忘记什么”
- ✅ 在机器翻译、语音识别等任务表现优异

**缺点**：
- ❌ 结构复杂，参数量大（是RNN的4倍）
- ❌ 计算成本高，训练速度慢
- ❌ 仍然存在一定的梯度消失问题

**适用场景**：机器翻译、文本生成、语音识别、情感分析

**代码示例**：
```python
lstm = nn.LSTM(input_size=5, hidden_size=7, num_layers=1, batch_first=True)
output, (hn, cn) = lstm(x)  # hn: 隐藏状态, cn: 细胞状态
```

---

<a id="gru"></a>
### 3. GRU (Gated Recurrent Unit)
**核心思想**：LSTM的简化版，将遗忘门和输入门合并为**更新门**，减少参数量。

**两大门控**：
| 门控 | 作用 | 公式 |
|------|------|------|
| **更新门 (Update Gate)** | 控制保留多少旧信息和加入多少新信息 | `z_t = σ(W_z · [h_{t-1}, x_t] + b_z)` |
| **重置门 (Reset Gate)** | 控制忽略多少过去的信息 | `r_t = σ(W_r · [h_{t-1}, x_t] + b_r)` |

**隐藏状态更新**：
- `h̃_t = tanh(W · [r_t * h_{t-1}, x_t] + b)` （候选隐藏状态）
- `h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t` （最终隐藏状态）

**优点**：
- ✅ 结构更简单，参数量少（是LSTM的3/4）
- ✅ 训练速度比LSTM快
- ✅ 在许多任务上效果与LSTM相当
- ✅ 同样能捕捉长距离依赖

**缺点**：
- ❌ 在某些需要精细控制的复杂任务上略逊于LSTM
- ❌ 仍然无法完全并行化

**适用场景**：资源受限场景、实时系统、中等长度序列建模

**代码示例**：
```python
gru = nn.GRU(input_size=5, hidden_size=7, num_layers=1, batch_first=True)
output, hn = gru(x)  # hn: 隐藏状态
```

---

### RNN vs LSTM vs GRU 对比总结

| 对比维度 | RNN | LSTM | GRU |
|---------|-----|------|-----|
| **门控数量** | 无 | 3个（遗忘、输入、输出） | 2个（更新、重置） |
| **参数量** | 最少 | 最多（≈4x RNN） | 中等（≈3x RNN） |
| **长距离依赖** | ❌ 差 | ✅ 优秀 | ✅ 良好 |
| **训练速度** | 快 | 慢 | 中等 |
| **内存占用** | 低 | 高 | 中等 |
| **梯度消失** | 严重 | 缓解 | 缓解 |
| **推荐场景** | 短序列 | 复杂长序列任务 | 平衡性能与效率 |

> **选择建议**：
> - 简单任务/短序列 → **RNN**
> - 复杂任务/长序列/高精度要求 → **LSTM**
> - 资源受限/追求效率 → **GRU**
> - 现代NLP任务 → 优先考虑 **Transformer**

<a id="attention"></a>
## 注意力机制 (Attention Mechanism)

### 注意力机制核心概念

可以把注意力机制想象成在图书馆找资料：
- **H（Hidden States）**：每本书的完整原始信息（封面+摘要+目录+正文+作者…）
- **K（Keys）**：图书索引标签（分类号、关键词、主题词）。用来和读者的查询词 Q 做匹配，决定哪本书该被翻出来。
- **Q（Query）**：读者的查询请求（当前需要关注的信息）
- **V（Values）**：书籍正文内容。一旦匹配成功，读者实际阅读和吸收的是这部分信息。
- **C（Context）**：按相关度加权后拼出的“定制资料包”。

**计算流程**：
1. 计算 Q 和 K 的相似度分数
2. Softmax 归一化得到注意力权重
3. 使用权重对 V 加权求和得到上下文向量 C

---

### 三种注意力机制对比

| 对比维度 | **软性注意力 (Soft Attention)** | **硬性注意力 (Hard Attention)** | **加性注意力 (Additive Attention)** |
|---------|-------------------------------|-------------------------------|----------------------------------|
| **别名** | 全局注意力 / 确定性注意力 | 局部注意力 / 随机性注意力 | Bahdanau 注意力 |
| **权重计算** | 对所有位置计算注意力权重 | 只关注少数关键位置（随机采样） | 通过线性层计算相似度 |
| **可导性** | ✅ **完全可导**，端到端训练 | ❌ **不可导**，需强化学习（RL）训练 | ✅ **完全可导**，端到端训练 |
| **计算方式** | `softmax(Q·K^T)` 或 `Linear(cat(Q,K))` | 基于概率采样选择位置 | `v^T tanh(W₁Q + W₂K)` |
| **梯度传播** | 所有位置都有梯度，平滑更新 | 只有被选中的位置有梯度 | 所有位置都有梯度，平滑更新 |
| **训练难度** | 简单，标准反向传播 | 困难，需要REINFORCE等算法 | 简单，标准反向传播 |
| **计算复杂度** | O(n)，需计算所有位置 | O(1)或O(k)，只计算少量位置 | O(n)，需计算所有位置 |
| **稳定性** | ✅ 稳定，收敛性好 | ❌ 不稳定，方差大 | ✅ 稳定，收敛性好 |
| **并行化** | ✅ 易于并行 | ❌ 难以并行（串行采样） | ✅ 易于并行 |
| **代表模型** | Transformer, Luong Attention | Show, Attend and Tell (图像字幕) | Bahdanau et al. (2014 NMT) |

---

### 1. 软性注意力 (Soft Attention)

**核心思想**：对输入序列的**所有位置**计算注意力权重，权重和为1，是确定性的连续函数。

**计算公式**：
```
attn_scores = Q · K^T  (点积注意力)
或
attn_scores = Linear(cat(Q, K))  (加性注意力)
attn_weights = softmax(attn_scores)
context = attn_weights · V
```

**优点**：
- ✅ 完全可导，可以使用标准的反向传播算法训练
- ✅ 训练稳定，收敛速度快
- ✅ 能够利用所有输入信息，不会遗漏重要特征
- ✅ 易于实现和调试

**缺点**：
- ❌ 计算量大，需要处理所有位置（序列很长时效率低）
- ❌ 可能会关注到无关紧要的位置（噪声干扰）
- ❌ 缺乏"聚焦"能力，不够稀疏

**适用场景**：
- 机器翻译（Transformer）
- 文本分类、情感分析
- 大多数现代NLP任务

---

### 2. 硬性注意力 (Hard Attention)

**核心思想**：每次只**随机选择**输入序列中的一个或少数几个位置进行关注，是离散的选择过程。

**计算方式**：
```
# 基于概率分布采样位置
position ~ Categorical(attn_weights)
context = V[position]  # 只获取选中位置的值
```

**优点**：
- ✅ 计算效率高，只处理少量位置
- ✅ 具有"聚焦"能力，更 interpretable（可解释）
- ✅ 可以学习到更稀疏、更有意义的注意力模式
- ✅ 适合资源受限场景

**缺点**：
- ❌ **不可导**，无法直接使用反向传播
- ❌ 需要使用强化学习（如REINFORCE算法）或Gumbel-Softmax技巧
- ❌ 训练不稳定，方差大，收敛困难
- ❌ 可能错过重要信息（采样偏差）

**训练技巧**：
- **REINFORCE算法**：将注意力选择视为动作，使用策略梯度
- **Gumbel-Softmax**：用连续松弛近似离散采样，实现近似可导
- **课程学习**：先训练软注意力，再迁移到硬注意力

**适用场景**：
- 图像字幕生成（Show, Attend and Tell）
- 视觉问答（VQA）
- 需要明确"看哪里"的任务

---

### 3. 加性注意力 (Additive Attention)

**核心思想**：通过一个**前馈神经网络**（线性层）来计算 Query 和 Key 之间的相似度，属于软性注意力的一种具体实现方式。

**计算公式**：
```
# Bahdanau 注意力公式
# 第一步：Q和K拼接后经过线性层得到相似度分数
attn_scores = v^T · tanh(W₁·Q + W₂·K + b)  # 或 Linear(cat(Q, K))
# 第二步：Softmax归一化得到注意力权重
attn_weights = softmax(attn_scores)
# 第三步：权重与V相乘得到上下文向量
context = attn_weights · V
```

**代码实现**（你的 `11-加性注意力.py`）：
```python
# 第一步：拼接 Q 和 K（在特征维度上拼接）
q_k_cat = torch.cat([query, key], dim=-1)  # (batch, seq_len, query_size+key_size)
# 第二步：通过线性层计算相似度分数（关键步骤：拼接→线性层→权重）
attn_scores = self.attn(q_k_cat)  # Linear层，输出维度=seq_len
# 第三步：Softmax 归一化得到注意力权重
attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len, seq_len)
# 第四步：加权求和得到上下文向量
attn_c = torch.bmm(attn_weights, value)  # (batch, seq_len, hidden_size)
```

##### 别忘了"融合层" attn_combine——它才是衔接下游 RNN 的关键

[NLP/11-加性注意力.py](NLP/11-加性注意力.py) 里的 `MyAttn` 类有**两个**线性层,容易看漏第二个:

| 线性层成员 | 第几行 | 输入 → 输出 | 用途 |
|-----------|-------|------------|------|
| `self.attn` | 第 26-27 行 | `query_size + key_size` → `seq_len` | 算**注意力分数** `attn_scores`(只用一次) |
| `self.attn_combine` | 第 33-34 行 | `query_size + hidden_size` → `output_size` | **把 Q 和动态 C 融合**,产出下游 GRU/LSTM 的 `input_x` |

```python
# 紧接上面的 attn_c, 第二个线性层在这里登场:
attn_q_c = torch.cat([query, attn_c], dim=-1)        # 把"原问题"和"答案上下文"拼一起
input_x  = self.attn_combine(attn_q_c)               # 线性融合 → 下一步 GRU 的输入
output, hn = self.gru(input_x)                       # 这才是加性注意力的"完整链路"终点
```

> 🌰 **生活类比**:`attn_c`(动态 C)只是"你查到的资料",**还要把"原问题 Q"和"资料 C"一起交给下游 RNN**——这步合并就是 `attn_combine` 干的事。源码注释第 19 行写得直白:"q 和动态 c 融合后的维度数 = 后续 rnn/lstm/gru/transformer 的输入维度"。

> ⚠️ **常见误解**:很多人以为加性注意力到 `attn_c` 就结束了。**没有**——少了 `attn_combine`,下游 RNN 拿不到原问题信息,效果会差一截。这一点在缩放点积注意力(乘性)里被简化掉了,这是两类注意力**最大的工程差别**。

**优点**：
- ✅ 完全可导，训练简单
- ✅ 能处理 Q 和 K 维度不同的情况（通过线性层映射）
- ✅ 在维度较小时表现比点积注意力更稳定
- ✅ 表达能力强，可以学习复杂的非线性关系

**缺点**：
- ❌ 计算速度较慢（相比点积注意力）
- ❌ 参数量较多（需要额外的权重矩阵 W₁, W₂, v）
- ❌ 空间复杂度高

**与乘性注意力的核心区别**：

**加性注意力**：
1. **Q和K拼接**：`cat(Q, K)` 在特征维度拼接
2. **经过线性层**：`Linear(cat(Q, K))` 得到相似度分数 `attn_scores`
3. **Softmax归一化**：得到注意力权重 `attn_weights = softmax(attn_scores)`
4. **计算上下文向量**：`C = attn_weights · V`
5. **再将 Q 和 C 拼接**：通过线性层融合得到最终输出 `output = Linear(cat(Q, C))`
6. 这个融合后的结果才是后续 RNN/LSTM/GRU 的输入

**乘性注意力**：
1. **计算点积**：`attn_scores = Q · K^T`
2. **缩放（关键步骤）**：`attn_scores = attn_scores / √d_k` （防止梯度消失，其中 `d_k` = Q/K 的特征维度）
3. **Softmax归一化**：`attn_weights = softmax(attn_scores)`
4. **计算上下文向量**：`C = attn_weights · V`
5. **C 就是最终输出**，不需要再与 Q 融合

> ⚠️ **重要说明**：现代乘性注意力几乎都使用**缩放点积注意力（Scaled Dot-Product Attention）**，即除以 `√d_k`（`d_k` 是 Query 和 Key 的特征维度，例如 64、128、512 等）。这是 Transformer 论文提出的关键技术，用于解决高维空间点积值过大导致 Softmax 梯度消失的问题。

| 特性 | 加性注意力 | 乘性注意力（点积） |
|------|----------|------------------|
| **相似度计算** | `Linear(cat(Q, K))` | `Q · K^T / √d_k` ← **缩放点积** (`d_k`=特征维度) |
| **上下文向量 C** | `C = attn_weights · V` | `C = attn_weights · V` |
| **最终输出** | `Linear(cat(Q, C))` ← **Q与C融合** | `C` ← **直接使用C** |
| 速度 | 较慢（多一步融合） | **更快**（矩阵乘法优化好） |
| 参数 | 需要额外参数（融合层） | 无需额外参数 |
| 维度要求 | Q、K维度可不同 | Q、K维度必须相同 |
| 数值稳定性 | 更稳定 | **需缩放**（否则高维时梯度消失） |
| 代表模型 | Bahdanau NMT (2014) | Transformer (2017) |

**适用场景**：
- 早期神经机器翻译系统
- Q 和 K 维度不一致的场景
- 序列长度较短的任务

---

### 4. 缩放点积注意力 (Scaled Dot-Product Attention)

**为什么需要缩放？**

当 Q 和 K 的维度 `d_k`（特征维度，如 64、128、512）很大时，点积 `Q · K^T` 的值会变得非常大，导致 Softmax 函数的梯度变得极小（接近于 0），从而引发**梯度消失问题**，使模型难以训练。

**数学原理**：
- 假设 Q 和 K 的元素是均值为 0、方差为 1 的独立随机变量
- 那么点积 `q · k` 的均值为 0，方差为 `d_k`（`d_k` = 特征维度）
- 当 `d_k` 很大时，点积值的分布范围会很广
- Softmax 在输入值很大时会进入饱和区，梯度接近 0

**解决方案**：
```python
# Transformer 的缩放点积注意力公式
# d_k = Query/Key 的特征维度（例如 64、128、512）
attn_scores = Q · K^T / √d_k  # 除以根号下维度
attn_weights = softmax(attn_scores)
output = attn_weights · V
```

**代码实现**（PyTorch）：
```python
import torch
import math

def scaled_dot_product_attention(query, key, value):
    """
    query: (batch, seq_len_q, d_k)  # d_k = 特征维度
    key:   (batch, seq_len_k, d_k)  # d_k = 特征维度
    value: (batch, seq_len_k, d_v)
    """
    d_k = query.size(-1)  # 获取特征维度 d_k
    
    # 计算点积
    attn_scores = torch.bmm(query, key.transpose(-2, -1))  # (batch, seq_len_q, seq_len_k)
    
    # 缩放（关键步骤！除以根号下特征维度 d_k）
    attn_scores = attn_scores / math.sqrt(d_k)
    
    # Softmax 归一化
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # 加权求和
    output = torch.bmm(attn_weights, value)  # (batch, seq_len_q, d_v)
    
    return output, attn_weights
```

**优点**：
- ✅ 解决了高维空间的梯度消失问题
- ✅ 训练更稳定，收敛更快
- ✅ 计算效率高（仍然是矩阵乘法）
- ✅ 成为现代 NLP 的标准配置

**为什么不用于加性注意力？**
- 加性注意力通过线性层和 tanh 激活函数已经将值限制在一定范围内
- 不会出现点积那样的数值爆炸问题
- 因此不需要额外的缩放操作

**实际应用**：
- **Transformer**：所有注意力头都使用缩放点积注意力
- **BERT、GPT、T5** 等预训练模型
- 几乎所有现代的基于 Attention 的架构

---

### 选择建议

| 场景 | 推荐注意力类型 |
|------|--------------|
| **现代NLP任务**（长序列、高精度） | 软性注意力（乘性/点积）+ Scaled |
| **资源受限/需要可解释性** | 硬性注意力（配合RL训练） |
| **Q/K维度不同** | 加性注意力 |
| **追求计算效率** | 乘性注意力（Transformer风格） |
| **小规模数据集** | 加性注意力（更稳定） |

> **发展趋势**：
> - 2014-2016：加性注意力主导（Bahdanau, Luong）
> - 2017至今：乘性注意力主导（Transformer）
> - 硬性注意力：特定场景使用（图像+文本多模态任务）

---

<a id="transformer"></a>
## Transformer 完整架构详解（NLP/12~18 文件）

> 📂 文件来源：[NLP/13~18.py](NLP/) + [encoder.py](NLP/encoder.py) / [decoder.py](NLP/decoder.py) / [input.py](NLP/input.py)，配套案例 [12.1-英译法案例.py](NLP/12.1-英译法案例.py)。

Transformer 是 2017 年 Google 论文 *Attention Is All You Need* 提出的架构，**抛弃 RNN，纯靠注意力**，奠定了 BERT/GPT/ChatGPT 的基石。

### 整体架构图（极简版）

```
输入文本 ──► [Embedding] ──► [+ 位置编码] ──┐
                                          ▼
                                  ┌─ Encoder × 6 ─┐
                                  │ ① 多头自注意力 │
                                  │ ② Add & Norm  │
                                  │ ③ 前馈网络    │
                                  │ ④ Add & Norm  │
                                  └────────┬─────┘
                                           │ 编码结果(K, V)
                                           ▼
                              ┌─ Decoder × 6 ────────────┐
目标文本──►[Embedding+位置]──►│ ① Masked 自注意力 + Norm  │
                              │ ② 跨注意力(Q from 解码器)│
                              │ ③ 前馈网络 + Norm         │
                              └──────────┬───────────────┘
                                         ▼
                          [Linear → log_softmax] ──► 词表概率
```

### 关键超参（Transformer-base）

| 参数 | 值 | 含义 |
|------|-----|------|
| `d_model` | 512 | 词向量/隐层维度 |
| `num_heads` | 8 | 多头数 |
| `d_k` | 64 | 每头维度 = `d_model / num_heads` |
| `d_ff` | 2048 | 前馈网络隐层 |
| `N` | 6 | 编码器/解码器层数 |
| `max_len` | 60~512 | 最大序列长度 |

---

<a id="positional-encoding"></a>
### 1️⃣ 输入层（[13-input编码器之位置编码.py](NLP/13-input编码器之位置编码.py)）

#### 词嵌入层

```python
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)  # ✨缩放
```

**为什么乘 `√d_model`？**
- 后面 attention 又要除 `√d_k`，先乘后除保持方差一致
- 放大 embedding 数值，避免被位置编码"淹没"

> 🌰 **生活类比**：embedding 是音量大小，位置编码是节拍器。如果音量太小，节拍器声音会盖住歌词，所以先把音量放大。

#### 位置编码（核心公式）

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

- `pos`：词在句子中的位置（0, 1, 2, ...）
- `2i / 2i+1`：词向量的偶数/奇数维度

**为什么要位置编码？**
Transformer 抛弃了 RNN 的"顺序读"，所有词同时输入；不告诉模型"这是第几个词"它就分不清"我吃饭"和"饭吃我"。

**为什么用 sin/cos？**
- 任何位置的编码都是固定的，不需要学
- 周期性 → 能泛化到训练时没见过的更长序列
- 三角函数性质 → 相对位置容易计算

> 🌰 **生活类比**：电影院给每个座位贴一个独一无二的座位号（位置编码），即使你蒙眼随便坐也能知道自己在第几排第几座。

#### 输入 = embedding + 位置编码

```python
x = embedding(x) * sqrt(d_model)
x = x + positional_encoding[:, :x.size(1)]
x = dropout(x)
```

---

<a id="mask"></a>
### 2️⃣ 掩码（[14-input编码器之mask掩码.py](NLP/14-input编码器之mask掩码.py)）

两种 mask 都是 0/1 矩阵，**0 的位置会被替换为 `-inf`**，softmax 后变 0。

#### Padding Mask（编码器+解码器都用）****

屏蔽 `[PAD]` 占位符，避免模型把"无意义填充"当成有效信息。
```python
padding_mask = (input_ids != 0).unsqueeze(-2)  # [batch, 1, seq_len]
```

> 🌰 **生活类比**：考试卷子上有些空白格不算分，阅卷老师直接跳过。

#### Causal Mask / Subsequent Mask（仅解码器）

下三角矩阵，让模型在预测第 t 个词时**只能看到前 t-1 个词**。
```python
mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角=1，上三角=0
# [[1,0,0,0],
#  [1,1,0,0],
#  [1,1,1,0],
#  [1,1,1,1]]
```

> 🌰 **生活类比**：写作文不能偷看后面的答案；翻译时第 3 个词只能看自己已写的前 2 个词。

---

<a id="encoder"></a>
### 3️⃣ 编码器（[15-transform之encoder.py](NLP/15-transform之encoder.py) + [16-层标准化.py](NLP/16-transform之层标准化.py)）

#### 缩放点积注意力（一次计算）

```python
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 缩放
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = softmax(scores, dim=-1)
    return weights @ V, weights
```

<a id="multi-head-attention"></a>
#### 多头注意力 Multi-Head Attention（[15-transform之encoder.py](NLP/15-transform之encoder.py)）

> 📌 **学习路线定位**：你已经学完了"[加性注意力](#attention)"和上一节的"缩放点积注意力"——单次 attention 怎么算已经清楚了。本节要讲的多头注意力，**就是把单次 attention 复制 8 份并行做，最后拼起来**，是 Transformer 编码器/解码器里真正在用的注意力。

##### 一句话定义

> **多头注意力 = 把 d_model 维的 Q/K/V 切成 head 段，每段独立做一次缩放点积注意力，最后再拼回 d_model 维。**

类比就是：原来一个老师（单头）打一份卷子的分；现在叫 8 个老师（8 头）各自打分，最后把 8 份评语合订成一份总报告。

---

##### 1. 为什么要多头？（动机）

单头注意力有个显而易见的问题：**一份注意力权重只能表达一种"关注模式"**。可一个句子里词与词的关系，从来不止一种——

| 关注角度 | 例：「我 昨天 在 公园 喂 鸽子」 |
|---------|--------------------------------|
| 谁是动作的发起者 | 「我」 ←→ 「喂」 |
| 动作的对象 | 「喂」 ←→ 「鸽子」 |
| 时间修饰 | 「昨天」 ←→ 「喂」 |
| 地点修饰 | 「在公园」 ←→ 「喂」 |

如果只用 1 个头，模型必须在这些关系之间"二选一"，注意力被稀释。把 d_model=512 拆成 8 头后，**每个头有自己独立的 W_q / W_k / W_v 线性层**（权重不共享！），可以**各自学一种关注模式**，互不干扰。

> 🌰 **生活类比**：单头 = 一个全科老师批卷子，又要看语法又要看立意，眼花缭乱；多头 = 8 个学科老师分头批，语法老师只看语法、立意老师只看立意，最后把 8 份评语合订——精度高、还能并行。

> ⚠️ **常见误解**：多头**不是**把同一个 attention 算 8 次取平均。每个头用的是**完全不同**的 W_q/W_k/W_v 参数（[15](NLP/15-transform之encoder.py) 第 110-111 行的 `clones()` 用 `copy.deepcopy` 做的就是这件事——4 个线性层互相独立、权重不共享）。

---

##### 2. 核心思想三步走（总览图）

无论代码看起来多复杂，**多头注意力永远只做这 3 件事**：

```
                    输入 x: [batch=2, 句长=4, d_model=512]
                                    │
                                    ▼
   ┌──────────── 第①步：线性变换 + 拆头 ────────────┐
   │  3 个独立线性层 W_q / W_k / W_v 各做一次 [512→512]  │
   │  把 512 维"切成"8 段，每段 d_k = 512/8 = 64 维     │
   │                                                    │
   │   x ──► W_q ──► Q [2,4,512] ─view─► [2,4,8,64]     │
   │   x ──► W_k ──► K [2,4,512] ─view─► [2,4,8,64]     │
   │   x ──► W_v ──► V [2,4,512] ─view─► [2,4,8,64]     │
   │                                                    │
   │   ─transpose(1,2)─► Q/K/V: [2,8,4,64]              │
   │                       ↑   ↑                        │
   │                   把"头"放到第二维，                │
   │                   让最后两维 (4, 64) 参与运算       │
   └────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌──────── 第②步：每头独立做缩放点积注意力 ────────┐
   │   8 个头并行计算（已经学过的 attention 函数）：    │
   │       scores = Q @ K^T / √d_k        [2,8,4,4]    │
   │       weights = softmax(scores)       [2,8,4,4]   │
   │       out_heads = weights @ V         [2,8,4,64]  │
   └────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌──────────── 第③步：拼接多头 + 输出投影 ───────────┐
   │   先 transpose(1,2) 把"头"放回第三维：[2,4,8,64]   │
   │   再 contiguous().view 把 8 头拼回 512 维：[2,4,512]│
   │   最后过第 4 个线性层 W_o (512→512) 融合各头信息   │
   │                                                    │
   │           输出: [2, 4, 512]  ← 形状和输入一致      │
   └────────────────────────────────────────────────────┘
```

**为什么是 4 个线性层而不是 3 个？**

[15-transform之encoder.py](NLP/15-transform之encoder.py) 第 127 行 `self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)` 这里克隆了 **4 个**线性层：

| 第 N 个 | 角色 | 用在哪一步 |
|--------|------|-----------|
| linears[0] | **W_q** 把 x 投影成 Query | 第 ① 步：变换 query |
| linears[1] | **W_k** 把 x 投影成 Key | 第 ① 步：变换 key |
| linears[2] | **W_v** 把 x 投影成 Value | 第 ① 步：变换 value |
| linears[3] | **W_o** 输出投影 (output projection) | 第 ③ 步：拼接后再融合一次 |

> 🌰 **生活类比**：W_q/W_k/W_v 是 3 个翻译官，把同一段中文翻成 3 种"提问视角"；W_o 是总编辑，把 8 个老师的评语合并润色成最终报告。

---

##### 3. 代码逐段拆解（对照 [15-transform之encoder.py](NLP/15-transform之encoder.py)）

###### 3.1 `__init__`：准备 4 个线性层和超参

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout_p=0.1):
        super().__init__()
        # ① 必须能整除：512 % 8 == 0，才能均匀分头
        assert embedding_dim % head == 0, 'head不能被整除'

        # ② 每头维度 d_k = 512 // 8 = 64
        self.d_k = embedding_dim // head
        self.head = head

        # ③ clones() = 用 deepcopy 拷 N 份独立的线性层（权重不共享）
        # 4 份：W_q, W_k, W_v, W_o
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        self.attn = None                       # 保存权重分布，便于可视化
        self.dropout = nn.Dropout(p=dropout_p)
```

**`clones()` 是什么？** ([15-transform之encoder.py](NLP/15-transform之encoder.py) 第 110-111 行)

```python
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

注意是 `copy.deepcopy`——**深拷贝**。这意味着每个线性层的权重矩阵在内存中是**独立的副本**，训练时各自更新自己的参数，**互不影响**。这正是"多头能学不同关注模式"的物理基础。

###### 3.2 `forward`：5 步搞定一次多头注意力

```python
def forward(self, query, key, value, mask=None):
    # ─── 第 0 步：拿到 batch_size，后面 view 要用 ───
    batch_size = query.size()[0]

    # ─── 第 1 步：线性变换 + 分头 + 转置 ───
    # 一行列表推导式同时处理 Q/K/V 三路
    # [2,4,512] ─model(x)─► [2,4,512]
    #          ─.view(batch,-1,head,d_k)─► [2,4,8,64]
    #          ─.transpose(1,2)──────────► [2,8,4,64]
    query, key, value = [
        model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        for model, x in zip(self.linears, (query, key, value))
    ]

    # ─── 第 2 步：调用上一节的 attention() 函数 ───
    # 输入 4 维：[2,8,4,64]，前 2 维 (batch, head) 不参与矩阵乘法
    # 只有最后两维 (4, 64) × (64, 4) 在做点积
    # 输出: x = [2,8,4,64], self.attn = [2,8,4,4]
    x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

    # ─── 第 3 步：拼接多头（先 transpose 再 view） ───
    # [2,8,4,64] ─.transpose(1,2)─► [2,4,8,64]
    #            ─.contiguous()──► （拷贝出连续内存，view 才不会报错）
    #            ─.view(batch,-1,head*d_k)─► [2,4,512]
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

    # ─── 第 4 步：最后一个线性层 W_o 做融合 ───
    # linears[-1] 就是第 4 个线性层
    return self.linears[-1](x)
```

---

##### 4. 关键形状追踪表（背下来，看代码就一目了然）

以 `batch=2, 句长=4, d_model=512, head=8, d_k=64` 为例：

| 步骤 | 操作 | 形状变化 | 说明 |
|------|------|---------|------|
| 输入 | x | `[2, 4, 512]` | (句子数, 句长, 词维度) |
| ①.1 | `W_q(x)` `W_k(x)` `W_v(x)` | `[2, 4, 512]` | 3 个独立线性层各做一次投影 |
| ①.2 | `.view(2, -1, 8, 64)` | `[2, 4, 8, 64]` | 把最后一维 512 切成 (8 头 × 64 维) |
| ①.3 | `.transpose(1, 2)` | `[2, 8, 4, 64]` | **关键转置**：把"头"放到第二维 |
| ② | `attention(Q, K, V)` 内部 | | |
| ②.1 |   `Q @ K^T / √d_k` | `[2, 8, 4, 4]` | 句长 × 句长 的注意力分数 |
| ②.2 |   `softmax + @V` | `[2, 8, 4, 64]` | 加权求和后的"动态 c" |
| ③.1 | `.transpose(1, 2)` | `[2, 4, 8, 64]` | 把"头"转回第三维 |
| ③.2 | `.contiguous().view(2,-1,512)` | `[2, 4, 512]` | 8 头拼回 d_model |
| ③.3 | `W_o(x)` | `[2, 4, 512]` | 输出投影,形状回到原点 |

**两个值得注意的点**：

1. 中间最大维度是 **4 维** `[batch, head, seq_len, d_k]`——前两维 `(batch, head)` 是"独立赛道"，注意力运算只用最后两维 `(seq_len, d_k)`。
2. 输入和输出形状**完全一致**(`[2, 4, 512]`)——这就是为什么 Transformer 可以**堆 6 层**编码器层而无需任何 reshape。

---

##### 5. 三个最容易踩的坑

###### 坑 1：为什么 view 之后必须 `transpose(1, 2)`？

```python
.view(batch_size, -1, self.head, self.d_k)   # [2, 4, 8, 64]
.transpose(1, 2)                              # [2, 8, 4, 64]
```

矩阵乘法 `Q @ K^T` 永远只看**最后两维**。如果不转置，最后两维是 `(8, 64)`——这是"头数 × 每头维度"，**不是**我们想算的"句长 × 每头维度"，结果完全错。

转置后最后两维变成 `(4, 64)`：4 个词、每个 64 维——这才是注意力分数 `[2, 8, 4, 4]`("4 个词彼此互相关注")的正确来源。

> 🌰 **生活类比**：写信封先要把"收件人"放在最外面，邮差才知道往哪送。`transpose(1,2)` 就是把"句长这一维"挪到能被 `matmul` 看见的位置。

###### 坑 2：为什么拼接时必须先 `contiguous()` 再 `view()`？

```python
x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
```

PyTorch 的 `transpose` **不会真的搬数据**——它只是改了"如何读取这块内存"的元信息（步长 stride）。这导致内存在物理上**不连续**。而 `view()` 要求张量必须**内存连续**才能直接重塑形状，否则会抛出 `RuntimeError: view size is not compatible`。

`contiguous()` 的作用就是：**新开一块连续内存，把数据真正搬过去**，让后续 `view()` 能用。

> 💡 **替代方案**：`reshape()` 在内存不连续时会自动复制；但显式写 `contiguous().view()` 让代码意图更清楚——"我知道这里有性能开销，我接受"。

###### 坑 3：mask 在多头里为什么要 `unsqueeze(1).unsqueeze(2)`？

来自 [15-transform之encoder.py](NLP/15-transform之encoder.py) 第 205 行的测试代码：

```python
mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
#                                      ↑ 加 head 维     ↑ 加 query 维
# x:    [2, 4]   ──► 句子里非 0 的位置为 1
# .unsqueeze(1): [2, 1, 4]
# .unsqueeze(2): [2, 1, 1, 4]
# 广播到 scores 的形状: [2, 8, 4, 4]
```

**为什么是 4 维？** 因为多头里的 `scores` 已经是 4 维的 `[batch, head, seq_len_q, seq_len_k]`。mask 必须能对齐这个形状，才能用 `masked_fill` 把 padding 位置打成 `-inf`。

**广播规则**：

| mask 维度 | scores 维度 | 广播后 |
|----------|-------------|--------|
| `[2, 1, 1, 4]` | `[2, 8, 4, 4]` | `[2, 8, 4, 4]` ✅ 完美对齐 |
| 第 1 维 `1` → 广播到 `8`（所有头共用同一个 padding mask） | | |
| 第 2 维 `1` → 广播到 `4`（每个 query 词都看不见同样的 padding 位置） | | |

> 🌰 **生活类比**：班里 8 个老师批同一份卷子，"答题区外的空白格不给分"这条规则是**全班共用**的——所以 mask 只需要 1 份，靠广播分发到 8 个头身上。

---

##### 6. 自注意力 vs 一般注意力（源码注释里反复强调的核心区分）

[15-transform之encoder.py](NLP/15-transform之encoder.py) 文件开头第 7-8 行就给出了"判别式"：

```python
# q=k=v -> 自注意力
# q!=k=v -> 编码器-解码器一般注意力
```

**两者代码完全相同**——区别只在**调用时传什么参数**：

| 类型 | 在哪里用 | 调用方式 | 含义 |
|------|---------|---------|------|
| **自注意力** Self-Attention | 编码器每层、解码器第 1 个子层 | `mha(x, x, x, mask)` | 句子**内部**词与词互相关注（"我"看着"喂"，"喂"看着"鸽子"） |
| **跨注意力** Cross-Attention | 解码器第 2 个子层 | `mha(decoder_x, memory, memory, src_mask)` | 解码端词去**关注编码端**输出的 memory（翻译时"看原文") |

```python
# 编码器层（[15] EncoderLayer）：q = k = v = x（自己看自己）
x = self.sublayer1(x, lambda v: self.self_attn(v, v, v, mask))

# 解码器层（[17] DecoderLayer）的第 2 个子层：Q 来自解码器，K/V 来自编码器
x = self.sublayer2(x, lambda v: self.cross_attn(v, memory, memory, src_mask))
```

> 🌰 **生活类比**：
> - **自注意力** = 学生写作文时反复回看自己写的前文，确保上下文连贯
> - **跨注意力** = 学生做翻译题时一边看原文一边写译文，原文就是 memory

更详细的 Q/K/V 来源对照见 [编解码链路](#encoder-decoder-link) 章节。

---

##### 7. 三个高频深度问答(读懂这三问,多头注意力就真懂了)

###### Q1:`model(x).view(batch_size, -1, self.head, self.d_k)` 这一步是**复制**还是**折叠**?

**纯折叠,零拷贝。**

`view` 在 PyTorch 里**只改"如何读取这块内存"的元信息(shape/stride),不动底层数据**。

输入 `model(x)` 形状 `[2, 4, 512]`,内存里就是一长串 float:

```
token 0 的 512 个特征: [f0, f1, ..., f63 | f64, ..., f127 | ... | f448, ..., f511]
                       └──── 64 ────┘   └──── 64 ────┘    ...  └──── 64 ────┘
                          "头 0"            "头 1"     ...        "头 7"
```

`.view(2, -1, 8, 64)` 之后形状变成 `[2, 4, 8, 64]`——**底层 4096 个 float 一个都没动**,只是告诉 PyTorch:"以后请把这 512 个数当成 8 组 × 64 个来读"。

| 操作 | 是否复制内存 | 干什么 |
|------|-------------|-------|
| `view` | ❌ 不复制 | **折叠**:改 shape,共享 storage |
| `transpose` | ❌ 不复制 | 改 stride,但会让内存"不连续" |
| `contiguous` | ✅ **复制** | 开一块新内存把数据搬过来,让后续 view 能用 |

> 💡 **关键点**:8 个头的 `W_q` 实际上**藏在同一个 `nn.Linear(512, 512)` 矩阵里**——这 512×512 矩阵的前 64 列是头 0 的 W_q,接下来 64 列是头 1 的 W_q……反向传播时各自梯度独立,效果等价于 8 个独立小线性层,但**计算和存储都是一次完成**。这正是多头注意力**几乎没有额外开销**的物理原因。

###### Q2:每个头代表的语义信息**相同**还是**不同**?

**不同——而且这正是多头存在的意义。**

代码里 8 个头的机制完全一样(同一个 `attention(Q,K,V)` 公式,同样的形状),它们**唯一的区别**是 `W_q`/`W_k`/`W_v` 的权重不同——而权重不同来自两个原因:

1. **随机初始化不同**:`copy.deepcopy` 拷出 4 个独立线性层,PyTorch 默认用 Kaiming 均匀分布做随机初始化,8 个头的 64×64 切片初始权重就不一样
2. **梯度更新不同**:训练时每个头收到的反向梯度不同,各自收敛到不同的局部最优

最终各头**自然涌现**出不同的关注模式。学界对 BERT 等模型做过大量探针实验,典型(不绝对)的分工有:

| 头的类型 | 关注什么 |
|---------|---------|
| 位置头 | 当前词的左/右邻居 |
| 句法头 | 主谓/动宾依存关系 |
| 共指头 | 代词指向的先行词 |
| 标点头 | 句号/逗号(段落边界) |
| 稀有词头 | 重点关注低频词 |
| 冗余头 | 几乎不起作用,可以剪枝掉 |

> ⚠️ **重要纠偏**:这种"分工"是**事后探针解释**,**不是**代码规定的。源码里没有任何一行告诉模型"头 0 学语法、头 1 学情感"——是损失函数的优化压力 + 初始化的随机性,自然把 8 个头推向不同方向。

> 🌰 **类比**:8 个学生看同一篇课文(同一份输入),起点不同(初始权重不同),老师只布置了一份作业(同一个 loss)。最后每个学生自然擅长不同角度——有的对语法敏感,有的对情感敏感。

> 📚 **冷知识**:论文 *Are Sixteen Heads Really Better than One?* (Michel et al., 2019) 发现许多注意力头**剪枝掉也不损精度**,说明实际有效的头数往往少于设计数——这是后续模型压缩的一个研究方向。

###### Q3:既然每个头不完整,是不是必须**重新拼起来**才完整?

**是的,而且要分两层理解——形状层"必须拼",语义层"必须融"。**

**第一层:形状必须拼**

```
单头输出:  [batch=2, seq=4, d_k=64]   ← 64 维,喂不进下游 FFN
拼接之后:  [batch=2, seq=4, d_model=512]  ← 512 维,可以继续传递
```

下游(前馈网络、下一层 LayerNorm)都要求输入是 **512 维**。单个头的 64 维向量从工程上就**喂不进下一层**——必须拼。

**第二层:`W_o` 才是真正的"融合"——这步最容易被忽略**

`view` 拼接(第 173 行)只是把 8 个 64 维向量"贴在一起",**并没有让它们交互**:

```
view 拼接后:  [头0的64维 | 头1的64维 | ... | 头7的64维]   ← 8 段独立信息并排放着,彼此不通气
                              │
                              ▼  W_o (512→512)  ← 第 4 个线性层登场
W_o 融合后:  [混合了所有 8 个头视角的 512 维]            ← 每一维都是 8 头的加权和
```

`W_o` 这个 512×512 的输出投影矩阵的作用就是**让 8 个头的信息真正"对话"**:它的每一行权重决定了"输出的某一维应当从哪几个头各取多少"。`W_o` 也是可训练参数,它学到的是**"融合策略"**。

> ⚠️ **常见误解**:有人以为多头注意力是"8 个头各算各的,最后拼接就完事"。**不是**——`W_o` 的存在让多头变成"8 个头各自学一个视角,**再由 `W_o` 学怎么把这 8 个视角综合起来用**"。这正是为什么 [15-transform之encoder.py:127](NLP/15-transform之encoder.py#L127) 克隆的是 **4 个**线性层而不是 3 个——前 3 个负责"分头"(W_q/W_k/W_v),第 4 个负责"合头"(W_o),**分与合一一对应**。

> 🌰 **类比**:8 台监控摄像头各拍房间一个角度。每台摄像头的画面**形状完整**(都是合法视频),但**信息残缺**(单角度看不到其他)。`view` 把 8 路画面贴到监控墙上,`W_o` 是值班员——他根据 8 路画面**综合判断**"现在房间整体什么状态"。

###### 三问串联记忆

> **"`view` 是折叠不复制(物理零开销),8 头语义自学不互通(训练涌现),`W_o` 把残片融成全景(分头-合头一一对应)。"**

---

##### 8. 一句话记忆口诀

> **"3 路投影分 8 头，每头各做点积注意力，拼回 512 再投影一次。"**

更细一点的 5 步口诀（对照代码顺序）：

```
① 投   ─ W_q W_k W_v 把 x 投成 Q/K/V    [2,4,512]
② 切   ─ view 把 512 切成 8×64           [2,4,8,64]
③ 转   ─ transpose 把 head 放到第 2 维   [2,8,4,64]
④ 算   ─ attention(Q,K,V,mask)           [2,8,4,64]
⑤ 拼   ─ transpose+contiguous+view       [2,4,512]
⑥ 融   ─ W_o 融合多头结果                [2,4,512]
```

记住这 6 步加上"输入输出形状一样都是 `[B, L, d_model]`"，多头注意力的代码就再也不会让你迷茫了。

---

> 📝 **小结**：多头注意力是 Transformer 编码器的"核心引擎"。下一节的**前馈网络**是引擎旁边的"非线性增强器"，再配合本章后面的 **Add & Norm** 残差结构，就组成了完整的编码器层 EncoderLayer——把 EncoderLayer 堆 6 层，就是 Transformer 的 Encoder。


#### 前馈网络

```python
# 两层全连接 + ReLU
FFN(x) = Linear_2( ReLU( Linear_1(x) ) )
# 维度: 512 → 2048 → 512
```

> 🌰 **生活类比**：先把信息"展开"到大房间（2048）方便整理，再"压缩"回原房间（512）。

#### Add & Norm（残差 + 层归一化）（[16-transform之层标准化.py](NLP/16-transform之层标准化.py)）

```python
class SublayerConnection(nn.Module):
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
        # 等价于：LayerNorm(x + dropout(SubLayer(x)))
```

| 组件 | 作用 |
|------|------|
| **残差连接** `x + sublayer(x)` | 信息高速公路，防止深层网络梯度消失 |
| **LayerNorm** | 在每个 token 的特征维度归一化（≠ BatchNorm 在 batch 维度） |
| **Dropout** | 训练时随机丢弃，防过拟合 |

> 🌰 **生活类比**：
> - **残差** = 写作业时保留草稿，万一新答案错了还能回头看
> - **LayerNorm** = 每个学生自己量身高体重再标准化（个人内部）；BatchNorm 是全班一起标准化（不适合 NLP，因为句子长度不一）

##### LayerNorm 源码细节（[16-transform之层标准化.py](NLP/16-transform之层标准化.py) 第 271-307 行）

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # 可学习参数 a2(γ): 缩放系数, 初始化为 1
        self.a2 = nn.Parameter(torch.ones(features))
        # 可学习参数 b2(β): 偏移系数, 初始化为 0
        self.b2 = nn.Parameter(torch.zeros(features))
        # 防止分母为 0 的极小常数
        self.eps = eps

    def forward(self, x):
        # 在最后一维(词维度 d_model)上求均值和标准差
        # x: [2, 4, 512] → mean/std: [2, 4, 1]
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        # 标准化 + 仿射变换: y = a2 * (x - μ) / (σ + ε) + b2
        return self.a2 * (x - mean) / (std + self.eps) + self.b2
```

**三个关键点**：

| 项 | 说明 |
|----|------|
| `a2` / `b2` 为什么是 `Parameter`？ | 标记为**可学习参数**，反向传播时 `optimizer` 会更新它们。`a2` 学"该放大多少"，`b2` 学"该平移多少"。 |
| 为什么是 `dim=-1`？ | 最后一维是词维度 `d_model=512`。**每个 token 自己内部归一化**——这正是 LayerNorm 与 BatchNorm 的根本差别。 |
| 为什么要 `eps=1e-6`？ | 当某个 token 所有 512 个特征都相同时，`std=0`，分母为 0 会产生 `nan`。`eps` 是兜底安全垫。 |

> ⚠️ **LayerNorm vs BatchNorm 一句话记忆**：
> - **BatchNorm**：跨样本归一化(对一批句子的同一个特征求均值/方差)——**句子长度不一**时不适合
> - **LayerNorm**：跨特征归一化(对一个 token 的 512 个特征求均值/方差)——**与句子长度无关**,所以是 NLP 的默认选择

##### SublayerConnection 的三种写法（源码注释里其实写了三种）

[16-transform之层标准化.py](NLP/16-transform之层标准化.py) 第 380-390 行的注释列了 3 种 Add & Norm 顺序，**Transformer 论文用的是第 3 种**：

```python
# 方式1: norm(x) → sublayer → dropout + x         # Pre-LN, 训练最稳定(后来流行)
# 方式2: sublayer → norm → dropout + x            # 不推荐, 容易训练不稳定
# 方式3: norm(x + dropout(sublayer(x)))           # Post-LN, 论文原版 ← 本仓库实现
```

新手只需记住**方式 3**就够了。如果以后看到 BERT/GPT 源码用方式 1(Pre-LN)，那是后续工作的优化版,不影响理解。

#### 编码器层堆叠 6 次

```python
class EncoderLayer(nn.Module):
    def forward(self, x, mask):
        x = self.sublayer1(x, lambda v: self.self_attn(v, v, v, mask))  # 自注意力
        x = self.sublayer2(x, self.feed_forward)                         # 前馈
        return x

class Encoder(nn.Module):
    def __init__(self, layer, N=6):
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
```

---

<a id="decoder"></a>
### 4️⃣ 解码器（[17-transform-decoder.py](NLP/17-transform-decoder.py)）

解码器有**三个子层**（编码器只有两个）：

```python
class DecoderLayer(nn.Module):
    def forward(self, x, memory, src_mask, tgt_mask):
        # ① Masked 自注意力（看自己已写的）
        x = self.sublayer1(x, lambda v: self.self_attn(v, v, v, tgt_mask))
        # ② 跨注意力（Q 来自解码器，K/V 来自编码器输出 memory）
        x = self.sublayer2(x, lambda v: self.cross_attn(v, memory, memory, src_mask))
        # ③ 前馈
        x = self.sublayer3(x, self.feed_forward)
        return x
```

> 🌰 **生活类比**：写英译中
> - ① 看一眼自己刚写的几个字（自注意力 + causal mask）
> - ② 回头看看英文原文哪里还没翻（跨注意力，Q=自己的疑惑，K/V=英文原文）
> - ③ 整合一下，写下下一个汉字（前馈）

#### 三个子层逐行走读（[17-transform-decoder.py](NLP/17-transform-decoder.py) 第 27-44 行）

源码里 `forward` 的参数实际叫 `source_mask`(编码器侧的 padding mask) 和 `target_mask`(解码器侧"padding & causal" 合成 mask),README 上面代码块里写成 `src_mask`/`tgt_mask` 是简化别名,二者一一对应。

| 子层 | 调用代码 | Q | K | V | 用哪个 mask | 干什么 |
|------|---------|---|---|---|-----------|-------|
| ① 自注意力 | `sublayer[0](x, λx: self_attn(x,x,x, target_mask))` | 解码端 x | 解码端 x | 解码端 x | `target_mask` | "看自己已写的"——配合 causal mask 不偷看未来 |
| ② 跨注意力 | `sublayer[1](x, λx: src_attn(x,m,m, source_mask))` | 解码端 x | 编码器输出 `memory` | 编码器输出 `memory` | `source_mask` | "对照原文"——把翻译进度对齐到源句的非 padding 位置 |
| ③ 前馈 | `sublayer[2](x, feed_forward)` | — | — | — | — | 非线性变换,与编码器 FFN 完全相同 |

**两个 mask 的合成方式**(源码第 80-86 行):

```python
# target 侧的两类 mask 用按位与合成,只有都为 1 才算"可以看"
target_padding_mask = (target != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)   # 屏蔽 PAD
target_causal_mask  = torch.tril(torch.ones(4, 4)).unsqueeze(0).unsqueeze(0)       # 屏蔽未来词
target_mask = target_padding_mask & target_causal_mask                             # 两个都满足
```

> 💡 完整的 Q/K/V 来源、memory 怎么从编码器一路传到解码器、训练 vs 推理的差异,跳到 [#encoder-decoder-link](#encoder-decoder-link) 看"6️⃣ 编码器⇄解码器全链路"——本节只做单层走读,避免重复。

---

### 5️⃣ 输出层（[18-transform之output.py](NLP/18-transform之output.py)）

```python
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        # 源码里的属性名实际是 self.out（不是 self.proj）
        self.out = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        # 沿最后一维(词表维度)做 log_softmax
        return torch.log_softmax(self.out(x), dim=-1)
```

把 512 维隐藏向量映射回**词表大小**，再 log_softmax 得到每个词的概率对数（配合 NLLLoss 用）。

> 🌰 **生活类比**：从"我懂了什么意思"（隐藏向量）翻译成"该说哪个具体的词"（词表概率）。

##### FAQ:为什么是 `log_softmax` 而不是直接 `softmax`?

| 角度 | 解释 |
|------|------|
| **配合损失函数** | `log_softmax + nn.NLLLoss` ≡ `nn.CrossEntropyLoss`(详见 [交叉熵章节](#cross-entropy))。Transformer 训练时常用 NLLLoss,所以输出层先把 log 做掉。 |
| **数值稳定性** | softmax 涉及 `exp(x)`,当 logits 很大时容易溢出;`log_softmax` 内部用 "log-sum-exp" 技巧,数值更稳。 |
| **梯度更友好** | log 把 `(0, 1]` 的概率拉到 `(-∞, 0]` 的对数域,避免极小概率(如 1e-20)在反向传播时直接归零。 |

> ⚠️ **常见错误**:模型已经输出 `log_softmax` 了,**不要**再外面套 `nn.CrossEntropyLoss`(它内部又会做一次 log_softmax)——应该用 `nn.NLLLoss`。配套关系参见 README [损失函数与输出层选择表](#cross-entropy)。

如果做推理需要真正的概率分布,只需 `torch.exp(gen_result)` 把对数转回概率:

```python
predicted_indices = torch.argmax(gen_result, dim=-1)   # 取概率最大的词 ID(argmax 在 log 域和概率域结果一样)
probabilities = torch.exp(gen_result)                  # 需要展示概率值时再 exp 回来
```

---

<a id="encoder-decoder-link"></a>
### 6️⃣ 编码器 ⇄ 解码器 全链路联系（重点）

前面 5 节把 6 个组件单独讲了一遍，这一节回答最关键的问题：**它们之间到底怎么连？数据流到底长什么样？**

#### 6.1 数据流总览（带形状追踪）

设 `batch=2, src_len=tgt_len=4, d_model=512, num_heads=8, vocab=1000`：

```
源句 source: [2, 4]                    ← LongTensor，词 ID
   │
   ▼  Embeddings(vocab,512) × √512
[2, 4, 512]
   │
   ▼  + PositionalEncoding (broadcast [1, 4, 512])
[2, 4, 512]                            ← 编码器输入
   │
   ▼  Encoder × 6 层（自注意力 + FFN + Add&Norm）
[2, 4, 512] ★ memory                   ← 编码器最终输出（K, V 来源）
   │
   │           ┌─────────────────────────────┐
   ▼           ▼                             │
─────────────────────────────────            │
目标句 target: [2, 4]                         │
   │                                         │
   ▼  Embeddings × √512 + PE                 │
[2, 4, 512]                                  │
   │                                         │
   ▼  解码器子层① Masked 自注意力              │
   │   Q=K=V=自己, mask=tgt_mask             │
[2, 4, 512]                                  │
   │                                         │
   ▼  解码器子层② Cross-Attention ◄──────────┘  ← memory 进入
   │   Q=自己, K=V=memory, mask=src_mask
[2, 4, 512]
   │
   ▼  解码器子层③ 前馈网络
[2, 4, 512]
   │
   ▼  Decoder × 6 层（堆叠循环）
[2, 4, 512]
   │
   ▼  Generator: Linear(512→1000) + log_softmax
[2, 4, 1000]                            ← 每个位置在词表上的对数概率
```

#### 6.2 关键纽带：memory（编码器输出）

| 维度 | 说明 |
|------|------|
| 名称 | `memory`（也叫 `encoder_output`） |
| 形状 | `[batch, src_len, d_model]` |
| 含义 | 源句子每个词的"上下文增强表示"（每个词都已融合全局信息） |
| 用途 | 传入**每个**解码器层的 cross-attention，作为 K 和 V |
| 寿命 | 编码器跑一次后，整个解码过程都不变 |

> 🌰 **生活类比**：memory 像翻译考试时摆在桌上的英文原稿。无论你写到第几个汉字，都能随时回头看这份原稿；原稿不会变，永远只算一次。

#### 6.3 解码器三个子层的 Q/K/V 来源差异（最容易混的地方）

这是 Transformer 最精妙也最容易记混的一点：

| 子层 | Q 来自 | K 来自 | V 来自 | 用 mask | 作用 |
|------|--------|--------|--------|---------|------|
| ① Masked 自注意力 | 解码器当前层输入 | 同 Q | 同 Q | tgt_mask（padding+因果） | 看自己已生成的词 |
| ② Cross-Attention | ① 的输出 | **memory** | **memory** | src_mask（仅 padding） | 对齐到原文相关位置 |
| ③ 前馈 FFN | ② 的输出 | — | — | — | 非线性变换 |

代码对应（[17-transform-decoder.py](NLP/17-transform-decoder.py) 第 39~43 行）：
```python
# x = 解码器输入(目标词向量+位置编码), m = memory(编码器输出)
x = self.sublayer``[0]``(x, lambda x: self.self_attn(x, x, x, target_mask))   # ① Q=K=V=x
x = self.sublayer``[1]``(x, lambda x: self.src_attn(x, m, m, source_mask))    # ② Q=x, K=V=m
x = self.sublayer``[2]``(x, self.feed_forward)                                 # ③ FFN
```

> 🌰 **生活类比**（译者翻英文）：
> - 子层① = 看自己刚写的几个汉字，保持上下文连贯
> - 子层② = 抬头看英文原稿，找下一个该翻什么（**Q 是脑中疑问，K/V 是原稿**）
> - 子层③ = 大脑加工一下，准备下笔

#### 6.4 为什么 cross-attention 中 K 和 V 都来自 memory？

| 角色 | 含义 | 翻译场景类比 |
|------|------|------------|
| **Q (Query)** | "我现在要翻下一个词，需要原文哪部分？" | 译者的脑中疑问 |
| **K (Key)** | 原文每个词的"标签/索引卡"，用来匹配 Q | 原稿每个英文词的"索引卡" |
| **V (Value)** | 原文每个词的"实际内容" | 原稿英文词的"语义信息" |

K 和 V 都来自 **同一份** memory：
- K 用于"匹配/检索"——找到 Q 想要的位置
- V 用于"取值"——读出那个位置的实际信息
- 它们描述的是同一原文，所以共享 memory 是天经地义的（钥匙和保险柜里的钱必须对应）

而 Q 来自解码器，因为它代表**译者当前的需求**，每写一个词需求都不同。

#### 6.5 两种 mask 在编码器/解码器的分工

| Mask | 形状 | 编码器自注意力 | 解码器自注意力 | 解码器跨注意力 |
|------|------|--------------|---------------|---------------|
| **src_mask**（源 padding） | `[batch, 1, 1, src_len]` | ✅ 屏蔽源 PAD | ❌ | ✅ 屏蔽源 PAD |
| **tgt_mask**（目标 padding ∧ 因果） | `[batch, 1, tgt_len, tgt_len]` | ❌ | ✅ 屏蔽 PAD + 不看未来 | ❌ |

代码（[17-transform-decoder.py](NLP/17-transform-decoder.py) 第 78~86 行）：
```python
# 源 padding mask
source_mask = (source != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
# 目标 padding mask
target_padding = (target != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
# 目标因果 mask（下三角）
target_causal = torch.tril(torch.ones(4, 4)).type(torch.uint8).unsqueeze(0).unsqueeze(0)
# 目标 mask = padding ∧ causal
target_mask = target_padding & target_causal
```

> 🌰 **生活类比**：
> - src_mask = 原稿上有几个空格，提醒"这里别看"
> - tgt_mask = 自己写到一半，前面写过的字能看，后面没写的字不存在 + 空格也别看

#### 6.6 训练 vs 推理：数据流的根本差异

| 阶段 | 解码器输入 | 是否并行 | 编码器跑几次 |
|------|----------|---------|------------|
| **训练 (Teacher Forcing)** | 一次喂入完整目标句 `[BOS, y1, y2, y3]` | ✅ 全部时间步并行 | 1 次 |
| **推理 (自回归生成)** | 起始 `[BOS]` → 预测 → 拼接 → 再预测... | ❌ 必须串行 | 1 次（结果缓存） |

```
✅ 训练（全并行）：
  decoder([BOS, 我, 爱, 你], memory) ──► 一次输出 [我, 爱, 你, EOS]
                                          因果 mask 保证不偷看未来

⏳ 推理（自回归）：
  step1: decoder([BOS],          memory) → 预测 "我"
  step2: decoder([BOS, 我],       memory) → 预测 "爱"
  step3: decoder([BOS, 我, 爱],   memory) → 预测 "你"
  step4: decoder([BOS, 我, 爱, 你], memory) → 预测 EOS, 停止
```

> 🌰 **生活类比**：
> - 训练 = 老师给你完整答案，你按行抄但不能偷看后一行（因果 mask 让你只能看前面，所以可以并行抄完一整行）
> - 推理 = 边想边写，必须一字一字憋出来
> - 编码器只算一次 = 原稿摆桌上不动，省算力（这就是大模型 **KV-Cache** 优化的雏形）

#### 6.7 编码器内部一层的形状追踪（多头注意力示例）

```
x = pe_result                                    # [2, 4, 512]
  ↓ MultiHeadAttention(Q=K=V=x, mask=src_mask)
  ├─ Linear×3 (Q,K,V 各一份)                     # 3 个 [2, 4, 512]
  ├─ view + transpose: 拆 8 头                   # [2, 8, 4, 64]
  ├─ scores = Q @ Kᵀ / √64                       # [2, 8, 4, 4]
  ├─ softmax(scores.masked_fill(mask==0,-inf))   # [2, 8, 4, 4]
  ├─ scores @ V                                  # [2, 8, 4, 64]
  ├─ transpose + view: 合 8 头                   # [2, 4, 512]
  └─ 最终 Linear                                 # [2, 4, 512]
  ↓ Add & Norm                                   # [2, 4, 512]
  ↓ FFN: Linear(512→2048) → ReLU → Linear(2048→512)  # [2, 4, 512]
  ↓ Add & Norm                                   # [2, 4, 512]
× 6 层 → memory                                  # [2, 4, 512]
```

> 🌰 **生活类比**：
> - 拆 8 头 = 把 512 维"大议题"分给 8 个评委组，每组负责 64 维"小议题"
> - 合 8 头 = 把 8 组评委的意见拼起来，再过一道线性层"统一文书"

#### 6.8 完整 Transformer 一图记忆

```
源句 ─►[Embed+PE]─►Encoder×6─►memory ──┐
                                       │
目标句─►[Embed+PE]─►[自注意力(因果mask)]│
                          ↓            │
                  [跨注意力(Q=自己, KV=memory)] ◄──┘
                          ↓
                        [FFN]
                          ↓
                       × 6 层
                          ↓
                  [Linear→词表]
                          ↓
                  log_softmax → 词概率
```

#### 6.9 高频面试问答（Transformer 编解码器篇）

**Q1：解码器有几个注意力？分别叫什么？**
A：两个。① Masked Self-Attention（自注意力，看自己已写的）；② Cross-Attention / Encoder-Decoder Attention（跨注意力，Q 来自解码器，K/V 来自编码器输出 memory）。

**Q2：memory 是什么？为什么解码器每层都用同一个 memory？**
A：memory 是编码器最后一层的输出 `[batch, src_len, d_model]`，包含源句的完整上下文。每个解码器层都需要"对齐到原文"，所以都从同一份 memory 里取 K/V。这样设计也避免了重复计算。

**Q3：为什么训练时解码器可以并行，推理时却必须串行？**
A：训练时目标句完整可用，靠 **因果 mask** 保证第 i 个位置看不到 i+1 之后，所以可以一次性算完所有位置；推理时 i+1 位置的词还没生成，必须先算出 yi 才能拼接进去算 yi+1。

**Q4：如果删除 cross-attention 子层会怎样？**
A：解码器就完全看不到源句信息，相当于退化成纯语言模型（GPT 的结构）。这正是 GPT 系列采用"仅解码器"架构的原因——它做的是续写，不需要对齐到另一段文本。

**Q5：BERT 用编码器还是解码器？为什么？**
A：BERT 仅用编码器（双向自注意力，无因果 mask）。因为它做的是"理解类"任务（分类/抽取），需要双向看完整句；而 GPT 做"生成类"任务，必须单向（因果 mask）。Transformer 的编码器/解码器结构刚好对应这两类需求。

---

### 7️⃣ 英译法 Seq2Seq + 加性注意力实战（[12.1](NLP/12.1-英译法案例.py) / [12.2](NLP/12.2-英译法案例.py)）

这是 Transformer 出现**之前**的经典 NMT 架构，理解它能更好懂为什么 Transformer 是革命：

| 组件 | 实现 | 作用 |
|------|------|------|
| `EncoderRNN` | Embedding + GRU | 把英文句子编码成隐状态序列 |
| `AttnDecoderRNN` | Embedding + 加性注意力 + GRU + Linear | 一边解码一边对齐到原文 |
| Teacher forcing | 训练时用真值输入；按 step 线性衰减比例 | 加速收敛 + 防止误差累积 |

```python
# 加性注意力核心
attn_weights = softmax(self.attn(torch.cat([embed, hidden], dim=-1)))
context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
gru_input = self.attn_combine(torch.cat([embed, context], dim=-1))
```

#### 12.1 vs 12.2 两个文件的分工

| 文件 | 关键内容 | 主要函数/类 |
|------|---------|------------|
| [12.1-英译法案例.py](NLP/12.1-英译法案例.py) | **数据 + 模型搭建** | `normalizeString`(文本清洗)、`my_getdata`(读 eng-fra 双语对)、`MyPairsDataset`(Dataset)、`EncoderRNN`(Embedding + GRU 编码器)、`AttnDecoderRNN`(Embedding + 加性注意力 + GRU 解码器) |
| [12.2-英译法案例.py](NLP/12.2-英译法案例.py) | **训练 + 评估 + 可视化**(在 12.1 基础上追加) | `train_seq2seq` / `train_iters`(训练循环 + Teacher Forcing)、`seq2seq_evaluate`(自回归推理)、`dm_test_Attention`(注意力权重可视化) |

> 💡 **学习建议**:
> 1. 先把 12.1 跑通——只看清 `EncoderRNN.forward` 和 `AttnDecoderRNN.forward` 的形状变化即可
> 2. 再读 12.2 的 `train_iters`——重点看 **Teacher Forcing 比例**怎么随训练步数变化(用 `random.random() < teacher_forcing_ratio` 决定每一步喂"真值"还是喂"上一步预测")
> 3. 最后看 `seq2seq_evaluate`——推理时**没有真值可喂**,必须用上一步的输出当下一步输入(纯自回归)

> ⚠️ **Teacher Forcing 的本质**:训练时让解码器看到正确答案,加快收敛、防止误差累积;推理时拿不到答案,只能用自己的预测——**训练-推理的 gap 就是这里产生的**(Transformer 的 causal mask 也是为了让训练更接近推理)。详情对照 [#encoder-decoder-link](#encoder-decoder-link) 的 6.6 节"训练 vs 推理"。

#### Transformer vs Seq2Seq+Attention 对比

| 维度 | Seq2Seq+Attn | Transformer |
|------|-------------|------------|
| 主体 | RNN/GRU/LSTM | 全注意力 |
| 并行 | ❌ 必须串行 | ✅ 全部并行 |
| 长序列 | 易遗忘 | 更稳健 |
| 训练速度 | 慢 | 快 5~10 倍 |
| 当前主流 | 已被淘汰 | BERT/GPT/Claude 的祖宗 |

---

### Transformer 一句话记忆

> **"输入加位置，多头来注意，残差防梯消，前馈再过一遍，编完给解码，掩码盖未来，最后线性 + softmax 出词。"**

---



---

# 📖 资源与附录

> 本章为全文性的附录，与具体学科解耦：包含学习路线、API 速查、参考资料、常见问题。

<a id="learning-roadmap"></a>
## 学习路线建议

> 路线推荐按"先广再深"原则：先把每个阶段的核心概念和经典模型走一遍打底，再回头深挖某一块。下面是三阶段的最低必修清单。

### 机器学习阶段
1. ✅ 理解监督学习与无监督学习的区别
2. ✅ 掌握数据预处理（归一化/标准化）
3. ✅ 学习经典算法（KNN、线性回归、逻辑回归、K-Means）
4. ✅ 熟悉模型评估指标（准确率、精确率、召回率、F1、MSE等）

### 深度学习阶段
1. ✅ 掌握PyTorch基础操作（张量创建、形状变换、运算）
2. ✅ 理解激活函数的作用和选择
3. ✅ 学习神经网络基础（前向传播、反向传播、损失函数）
4. ✅ 实践自动微分和权重更新

### NLP阶段
1. ✅ 文本预处理（分词、去停用词、词性标注）
2. ✅ 词表示方法（One-Hot、Word2Vec [CBOW/Skip-gram]、Embedding）
3. ✅ RNN家族（RNN、LSTM、GRU）的原理和实现
4. ✅ 注意力机制（加性注意力、乘性注意力、缩放点积注意力）
5. ✅ 实战项目（英译法机器翻译）

---

## 常用API速查

> 需要"立刻能跑"的代码片段？本节按工具栈分类，复制粘贴即可使用。

### Scikit-Learn
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### PyTorch
```python
import torch
import torch.nn as nn

# 常用层
nn.Linear()      # 全连接层
nn.RNN()         # RNN层
nn.LSTM()        # LSTM层
nn.GRU()         # GRU层
nn.Embedding()   # 嵌入层

# 激活函数
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax()
```

### NLP工具
```python
import jieba
import jieba.posseg as pseg

# 分词
words = jieba.lcut("我爱自然语言处理")

# 词性标注
words = pseg.cut("我爱自然语言处理")
```

---

## 参考资料

- 📚 PDF文件夹：机器学习和深度学习预习资料
- 📖 NLP/资料：完整的NLP学习文档
- 💻 代码示例：每个知识点都有对应的Python实现

---

## 常见问题 FAQ

**Q1: 如何选择KNN的K值？**
A: 通常选择3-7，可以通过交叉验证选择最优K值。K值过小容易过拟合，过大容易欠拟合。

**Q2: 什么时候用标准化，什么时候用归一化？**
A: 大多数情况用标准化（Z-Score）。只有当数据有明显边界（如图像像素0-255）时用归一化。

**Q3: LSTM和GRU选哪个？**
A: 如果计算资源充足且任务复杂，选LSTM；如果追求效率和性能平衡，选GRU。两者效果通常相近。

**Q4: 注意力机制中为什么要缩放点积？**
A: 防止高维空间点积值过大导致Softmax梯度消失。除以√d_k可以将值拉回到合理范围。

**Q5: 如何处理类别不平衡问题？**
A: 
- 重采样（过采样少数类/欠采样多数类）
- 使用F1分数而非准确率评估
- 调整分类阈值
- 使用代价敏感学习

---

<a id="text-classification-project"></a>
# 🚀 文本分类项目实战 (THUCNews)

本节是整个项目的实战核心，按照 **传统机器学习 → 浅层神经网络 → 预训练大模型 → LLM大语言模型 → 模型压缩部署** 的脉络，循序渐进地完成中文新闻10分类任务。

## 项目演进路线图

```
01-data (数据准备 + EDA)
   ↓
02-rf (随机森林 + TF-IDF)        ← 传统机器学习基线
   ↓
03-fasttext (FastText)           ← 浅层神经网络，速度极快
   ↓
04-bert (BERT 微调)              ← 预训练大模型，效果最好
   ↓
05-LLM (DeepSeek API)            ← 零样本 / 少样本提示工程
   ↓
06-model-compression             ← 上线前的模型压缩
   ├── 量化 (Quantization)
   ├── 剪枝 (Pruning)
   └── 蒸馏 (Distillation)
```

> 💡 **生活类比**：这就像学厨师
> - 第1步：先认识食材（数据EDA）
> - 第2步：学家常菜（随机森林，老办法但稳定）
> - 第3步：学快餐（FastText，又快又好吃）
> - 第4步：学米其林大菜（BERT，工序复杂但味道惊艳）
> - 第5步：直接请大厨（LLM，告诉他要求即可）
> - 第6步：把大菜做成预制菜上线（模型压缩，让米其林大菜能进千家万户）

## 数据集：THUCNews 10分类

| 字段 | 说明 |
|------|------|
| 数据格式 | `文本\t标签ID` (UTF-8) |
| 类别数 | 10 |
| 类别 | finance / realty / stocks / education / science / society / politics / sports / game / entertainment |
| 数据划分 | train.txt / dev.txt / test.txt |
| 单条样本 | `中华女子学院：本科层次仅1专业招男生\t3` |

> 🌰 **生活类比**：就像把今日头条上 18 万条新闻标题按"财经/房产/股票/教育/科技/社会/政治/体育/游戏/娱乐"10 个频道分门别类。

---

<a id="data-eda"></a>
## 一、01-data：数据准备与EDA

### 1.1 EDA 是什么

**EDA (Exploratory Data Analysis，探索性数据分析)**：在建模前先"看一眼"数据长什么样。

> 🌰 **生活类比**：去菜市场买菜前，你得先逛一圈，看看今天什么菜新鲜、什么菜便宜、有没有不熟的菜。EDA 就是建模前的"逛菜市场"。

### 1.2 文本分类 EDA 关注哪些指标

| 指标 | 含义 | 用途 |
|------|------|------|
| **样本数量** | 训练/验证/测试集各多少条 | 决定 batch_size、epoch |
| **类别分布** | 每个类别有多少条 | 看是否类别不平衡 |
| **文本长度分布** | 句子长度的均值、最大值、95分位 | 决定 `padding_size` (常取 95~99 分位) |
| **词频统计** | 哪些词出现最多 | 找停用词、构建词表 |
| **缺失/重复** | 是否有空文本、重复样本 | 数据清洗依据 |

<a id="padding-size-selection"></a>
### 1.3 padding_size 怎么选

不能选最大长度（浪费算力），不能选平均值（一半文本被截）。
**推荐选 95% ~ 99% 分位数**，让 95% 以上的文本不被截断，同时保持高效。

> 🌰 **生活类比**：买衣服尺码不会按部门里最高的人买（浪费布），也不会按平均身高（一半人穿不下），而是按 P95（95% 的人能穿）。

---

<a id="rf-section"></a>
## 二、02-rf：随机森林 + TF-IDF（传统ML基线）

### 2.1 随机森林是什么

**随机森林 (Random Forest)** = 多棵决策树投票。

> 🌰 **生活类比**：让 100 个朋友帮你判断一条新闻是什么类别，每个朋友都给一个答案，最后少数服从多数。每个朋友（决策树）只看到部分特征（随机），所以判断角度不同，避免"群体盲点"。

<a id="tf-idf"></a>
### 2.2 TF-IDF：把文本变成数字

| 概念 | 公式 | 直观理解 |
|------|------|---------|
| **TF (词频)** | `某词在文档中出现次数 / 文档总词数` | "这个词在这篇文章里出现得多吗？" |
| **IDF (逆文档频率)** | `log(总文档数 / 包含该词的文档数)` | "这个词稀有吗？常见词权重低" |
| **TF-IDF** | `TF × IDF` | "对这篇文章而言，这个词重要吗？" |

> 🌰 **生活类比**：判断一篇文章是体育新闻 — "球"出现10次说明可能跟体育有关（高TF），但"的、是、了"出现100次也没用（低IDF，太普遍）。TF-IDF 就是同时考虑"出现多 + 区分度高"。

### 2.3 02-rf 模块流程

| 文件 | 作用 |
|------|------|
| [config.py](文本分类项目/02-rf/config.py) | 路径、超参数集中管理 |
| [dataEDA_Processing.py](文本分类项目/02-rf/dataEDA_Processing.py) | 加载文本 + jieba分词 + TF-IDF向量化 |
| [rf_train.py](文本分类项目/02-rf/rf_train.py) | 训练 RandomForestClassifier，保存 `.pkl` |
| [rf_predict_fun.py](文本分类项目/02-rf/rf_predict_fun.py) | 加载模型 + tokenizer，做单条预测 |
| [api.py](文本分类项目/02-rf/api.py) | FastAPI 提供 HTTP 推理接口 |
| [app.py](文本分类项目/02-rf/app.py) | Streamlit/Gradio 可视化界面 |

<a id="deployment-pipeline"></a>
### 2.4 上线四件套：训练 → 保存 → API → UI

这是工业界通用流程，每个项目都会重复：
1. **训练**：`fit()` 拟合数据
2. **保存**：`joblib.dump(model, "model.pkl")` 存盘
3. **API**：FastAPI 暴露 `/predict` 接口，输入文本返回类别
4. **UI**：网页让产品经理 / 用户自己测

> 🌰 **生活类比**：训练 = 学做菜；保存 = 把菜谱写下来；API = 在外卖平台开店；UI = 餐厅前台让顾客点菜。

---

<a id="fasttext-section"></a>
## 三、03-fasttext：FastText（浅层神经网络）

### 3.1 FastText 是什么

**FastText** 是 Facebook (Meta) 开源的文本分类工具，特点是 **简单 + 快 + 效果不错**。

核心思想：把句子里所有词的词向量取平均，再过一层全连接 + Softmax。

> 🌰 **生活类比**：判断一段话的情绪——把每个词的"情绪打分"全加起来求平均，再决定是开心还是难过。简单粗暴，但意外地有效。

### 3.2 FastText 三大杀手锏

| 特性 | 说明 | 好处 |
|------|------|------|
| **n-gram 子词** | 把"苹果"拆成"苹"、"果"、"苹果" | 处理未登录词（OOV）、错别字 |
| **层级 Softmax** | 用 Huffman 树替代普通 Softmax | 把分类速度从 O(N) 降到 O(logN) |
| **平均词向量** | 句向量 = 所有词向量的均值 | 速度极快，模型极小 |

### 3.3 FastText 训练数据格式

```
__label__sports 中华女子学院 本科 层次 仅 1 专业 招 男生
__label__science 苹果 发布 新款 iPhone 处理器
```

> ⚠️ **关键**：标签必须以 `__label__` 开头，词之间用空格分隔。

### 3.4 03-fasttext 模块流程

| 文件 | 作用 |
|------|------|
| [01-data_preprocess.py](文本分类项目/03-fasttext/01-data_preprocess.py) | 把 THUCNews 转成 FastText 格式 |
| [02-fasttext_word_2_auto.py](文本分类项目/03-fasttext/02-fasttext_word_2_auto.py) | 词级别训练 + 自动调参 |
| [02-fasttext_char_2_auto.py](文本分类项目/03-fasttext/02-fasttext_char_2_auto.py) | 字级别训练（中文常用，无需分词） |
| [predict_fun.py](文本分类项目/03-fasttext/predict_fun.py) | 加载 `.bin` 模型做预测 |

### 3.5 字级别 vs 词级别

| 维度 | 字级别 (char) | 词级别 (word) |
|------|-------------|--------------|
| 是否分词 | ❌ 不需要 | ✅ 需要 jieba |
| 词表大小 | 小（约5000字） | 大（几十万词） |
| OOV 问题 | 几乎没有 | 严重 |
| 推荐场景 | 中文、社交媒体短文本 | 英文、规范语料 |

> 🌰 **生活类比**：字级别 = 按字母拼写理解英文（unbreakable = un + break + able）；词级别 = 直接背单词（unbreakable）。中文用字级别更稳。

### 3.6 自动调参 `autotuneValidationFile`

FastText 提供"傻瓜式调参"：
```python
model = fasttext.train_supervised(
    input="train.txt",
    autotuneValidationFile="dev.txt",
    autotuneDuration=600  # 给它10分钟自己调
)
```

> 🌰 **生活类比**：你只告诉助理"我下周要面试，帮我搭配一套衣服"，他自己去试100套，最后给你最满意的那套。

---

<a id="bert-section"></a>
## 四、04-bert：BERT 微调（重点章节）

### 4.1 BERT 是什么

**BERT (Bidirectional Encoder Representations from Transformers)**：Google 2018 年提出的预训练语言模型。

核心思想：**先在海量无标注语料上做"完形填空"预训练，再在你的小数据集上微调**。

> 🌰 **生活类比**：BERT 就像一个读完了"百度百科+维基百科+网络小说+新闻"的大学生。他不知道你公司的具体业务，但语文功底好，你只需要"实习培训"几天，他就能上岗。这比从初中生（随机初始化）开始训练快得多、效果好得多。

<a id="bert-pretraining-tasks"></a>
### 4.2 BERT 的两大预训练任务

| 任务 | 全称 | 做什么 | 学会了什么 |
|------|------|--------|----------|
| **MLM** | Masked Language Model | 随机遮住15%的词，让模型猜 | 词的双向上下文 |
| **NSP** | Next Sentence Prediction | 判断两句话是不是连续 | 句子级别的关系 |

> 🌰 **生活类比**：MLM = 完形填空；NSP = 判断"这两段话是不是同一篇文章里相邻的两段"。

<a id="bert-three-ids"></a>
### 4.3 BERT 输入的三个 ID

输入文本会被 tokenizer 转成三类 ID：

| ID 类型 | 维度 | 含义 |
|---------|------|------|
| `input_ids` | [batch, seq_len] | 每个 token 在词表中的编号 |
| `attention_mask` | [batch, seq_len] | 1=真实token，0=padding填充 |
| `token_type_ids` | [batch, seq_len] | 句子A=0，句子B=1（NSP用） |

<a id="special-tokens"></a>
### 4.4 特殊 Token

| Token | ID | 作用 |
|-------|-----|------|
| `[CLS]` | 101 | 句首标记，**整句的语义都聚合到这个位置** |
| `[SEP]` | 102 | 句子分隔符 |
| `[PAD]` | 0 | 填充符 |
| `[UNK]` | 100 | 未登录词 |
| `[MASK]` | 103 | 完形填空标记 |

> 💡 **关键**：分类任务取 `[CLS]` 位置的输出向量（pooled output），因为它聚合了整句信息。

### 4.5 04-bert 项目结构

| 文件 | 作用 |
|------|------|
| [config.py](文本分类项目/04-bert/src/config.py) | 路径、`max_len`、`batch_size`、`learning_rate`、设备 |
| [utils.py](文本分类项目/04-bert/src/utils.py) | `BertDataset` + `build_dataloader()` |
| [bert_classifer_model.py](文本分类项目/04-bert/src/bert_classifer_model.py) | `BertModel` + `nn.Linear` 分类头 |
| [train.py](文本分类项目/04-bert/src/train.py) | 训练 + 验证 + 保存最佳模型 |
| [predict_fun.py](文本分类项目/04-bert/src/predict_fun.py) | 单条预测，含耗时统计 |

### 4.6 BertClassifier 模型结构

```python
class BertClassifier(nn.Module):
    def __init__(self):
        self.bert = BertModel.from_pretrained("bert-base-chinese")  # 12层Transformer
        self.fc = nn.Linear(768, 10)  # 分类头：768维 → 10类

    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(input_ids, attention_mask, return_dict=False)
        # pooled: [batch, 768]，[CLS]位置经过tanh的输出
        out = self.fc(pooled)  # [batch, 10]
        return torch.softmax(out, dim=1)
```

> 🌰 **生活类比**：BERT 是大脑（理解整句话），`fc` 是嘴巴（说出"这是什么类别"）。微调时大脑+嘴巴一起轻微调整，让它适应"新闻分类"这个具体岗位。

### 4.7 训练循环（必背模板）

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        logits = model(input_ids, attention_mask)        # 1. 前向
        loss = criterion(logits, labels)                  # 2. 算损失
        optimizer.zero_grad()                             # 3. 梯度清零
        loss.backward()                                   # 4. 反向传播
        optimizer.step()                                  # 5. 更新参数
    
    # 每个epoch在验证集评估，保存最佳模型
    f1 = evaluate(model, dev_loader)
    if f1 > best_f1:
        torch.save(model.state_dict(), save_path)
```

> 🌰 **生活类比**：5步PyTorch训练流程 = 学开车
> 1. 前向传播 = 看路 (输入→输出)
> 2. 算损失 = 发现自己偏了多少
> 3. 梯度清零 = 把上次的方向盘记忆清空
> 4. 反向传播 = 算出"应该往哪打方向盘"
> 5. 参数更新 = 真的去打方向盘

### 4.8 优化器选择

| 优化器 | 特点 | 何时用 |
|--------|------|------|
| SGD | 朴素梯度下降 | 大型CV任务、需精细调参 |
| Adam | 自适应学习率 | 通用首选 |
| **AdamW** | Adam + 解耦权重衰减 | **Transformer/BERT 标配** |

### 4.9 评估指标的 average 参数

```python
f1_score(y_true, y_pred, average='micro')  # 全局TP/FP/FN算一个F1
f1_score(y_true, y_pred, average='macro')  # 每类算F1后取平均
f1_score(y_true, y_pred, average='weighted')  # 按样本数加权平均
```

> 🌰 **生活类比**：班里10个小组比赛
> - micro = 把所有小组成绩混在一起算总分
> - macro = 每个小组算个平均分，再求平均（小组平等）
> - weighted = 大组权重大，小组权重小

---

<a id="llm-section"></a>
## 五、05-LLM：大语言模型（DeepSeek API）

### 5.1 为什么LLM能做分类

传统模型需要"训练→部署"，而 LLM 已经"读完了整个互联网"，**只需要写好提示词（Prompt），它就能直接做分类**——无需训练，这叫 **零样本（Zero-shot）** 或 **少样本（Few-shot）** 学习。

> 🌰 **生活类比**：传统ML是培养一个专科毕业生；LLM 是请一位读过万卷书的教授，你只需要告诉他要求，他就能直接答题。

### 5.2 Prompt 工程关键技巧（看 [deepseek_classifierLLM.py](文本分类项目/05-LLM/deepseek_classifierLLM.py)）

```python
system_prompt = '''
你是一个优秀的文本分类师...
参考案例：
文本：中国国家乒乓球队击败日本
类别：sports

备选类目：finance,realty,stocks,...,entertainment

请注意：
1. 仅能在【备选类目】中选择
2. 仿照案例分析
3. 模糊就选"拒识"
4. 回复格式：文本类别：xxx
'''
```

**Prompt 设计五要素**：
| 要素 | 作用 |
|------|------|
| **角色设定** | "你是一个优秀的文本分类师" — 让模型进入状态 |
| **任务说明** | 要做什么 |
| **少样本示例** | Few-shot — 给参考答案 |
| **约束条件** | 只能从给定类目选 / 不能编造 |
| **输出格式** | 限定返回格式，便于程序解析 |

> 🌰 **生活类比**：Prompt 就像点外卖的备注：
> - "微辣"（角色）
> - "麻辣香锅"（任务）
> - "上次那种"（示例）
> - "不要香菜"（约束）
> - "餐具单独放"（格式）

### 5.3 LLM 分类的优劣

| 优点 | 缺点 |
|------|------|
| ✅ 零训练成本，立刻能用 | ❌ 推理慢（一次几百ms~几秒） |
| ✅ 支持复杂指令 | ❌ API 调用要花钱 |
| ✅ 易于扩展类目 | ❌ 数据隐私问题 |
| ✅ 可解释（能给理由） | ❌ 输出格式不稳定，要做后处理 |

### 5.4 .env 文件管理密钥

```bash
# .env 文件（千万不要 git 提交！）
DEEPSEEK_API_KEY=sk-xxxxxxxx
base_url=https://api.deepseek.com
```

```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
```

> ⚠️ **安全规范**：API 密钥永远不要写进代码，要用环境变量；`.env` 必须加入 `.gitignore`。

---

<a id="compression-section"></a>
## 六、06-model-compression：模型压缩三板斧

### 为什么要压缩模型？

BERT-base 模型 ~400MB，单条推理 100~500ms。
**线上服务希望**：
- 模型小（手机/边缘设备能跑）
- 推理快（用户体验好）
- 算力省（GPU花费降低）

> 🌰 **生活类比**：米其林大菜（BERT）味道好但慢且贵；预制菜/速食包（压缩后的模型）味道接近，但 30 秒上桌、家家户户都能买。

### 三大压缩方法对比

| 方法 | 核心思想 | 体积压缩 | 速度提升 | 精度损失 | 难度 |
|------|---------|---------|---------|---------|------|
| **量化** | 把 float32 变 int8 | ~4x | 2~4x | <1% | ⭐ 简单 |
| **剪枝** | 把不重要的权重设为0 | 30~70% | 1~2x | 1~5% | ⭐⭐ 中等 |
| **蒸馏** | 用小模型学大模型 | 5~10x | 5~10x | 2~5% | ⭐⭐⭐ 复杂 |

<a id="quantization"></a>
### 6.1 量化 (Quantization)

**思路**：把模型权重的数据类型从 `float32`（32位浮点）变成 `int8`（8位整数）。

> 🌰 **生活类比**：
> - 量化前：写菜谱用"加 3.14159 克盐" → float32
> - 量化后：写菜谱用"加 3 克盐" → int8
> - 文件小了 4 倍，做出来的菜味道几乎一样。

#### 动态量化代码（看 [bert_model_quantization.py](文本分类项目/06-model-compression/bert_%20quantization/src/bert_model_quantization.py)）

```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # 只量化Linear层
    dtype=torch.qint8
)
```

#### 量化分类

| 类型 | 何时量化 | 精度 | 实现难度 |
|------|---------|------|---------|
| **动态量化** | 推理时实时量化激活值 | 高 | ⭐ 一行代码 |
| **静态量化** | 训练后用校准集预先量化 | 中 | ⭐⭐ 需校准 |
| **量化感知训练 (QAT)** | 训练时模拟量化 | 最高 | ⭐⭐⭐ 改训练 |

<a id="pruning"></a>
### 6.2 剪枝 (Pruning)

**思路**：神经网络里大部分权重接近 0，对结果影响很小，**直接把它们设为 0**。

> 🌰 **生活类比**：你衣柜里 70% 的衣服一年穿不到一次（权重接近0），把它们扔掉，衣柜清爽，找衣服更快，穿得几乎一样好。

#### 剪枝两大流派

| 类型 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **非结构化剪枝** | 按单个权重剪 | 灵活，精度高 | 需要稀疏矩阵库才能加速 |
| **结构化剪枝** | 按整个神经元/通道剪 | 真·加速 | 精度损失大 |

#### 全局非结构化剪枝代码（看 [bert_prune.py](文本分类项目/06-model-compression/bert_prune/bert_prune.py)）

```python
import torch.nn.utils.prune as prune

# 把 12 层 BERT 的 query 权重全局剪 30%
parameters_to_prune = [
    (model.bert.encoder.layer[i].attention.self.query, 'weight')
    for i in range(12)
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,  # 按L1范数（绝对值）剪
    amount=0.3                            # 剪掉30%
)

for module, param in parameters_to_prune:
    prune.remove(module, param)  # 永久应用剪枝
```

#### 稀疏度（Sparsity）

```python
sparsity = (权重为0的数量) / (权重总数)
```
比如剪了 30%，sparsity ≈ 0.3。

<a id="distillation"></a>
### 6.3 蒸馏 (Knowledge Distillation)

**思路**：训一个大模型（教师），让它把"知识"传给一个小模型（学生）。

> 🌰 **生活类比**：教师 = 高薪请来的特级教师；学生 = 实习老师。让特级教师上课，实习老师在旁边学。最后实习老师能 80% 还原特级教师的水平，但工资只要 1/10。

#### 教师 vs 学生（项目实例）

| 模型 | 大小 | 角色 |
|------|------|------|
| **BertClassifier** | ~400MB, 110M参数 | 教师 |
| **BiLSTMClassifier** | ~5MB, 几百万参数 | 学生 |

#### 蒸馏两种方式

| 方式 | 学习目标 | 公式 |
|------|---------|------|
| **硬标签蒸馏** | 学生学习教师的**预测类别** | `loss = CE(student_logits, teacher_argmax)` |
| **软标签蒸馏** | 学生学习教师的**概率分布** | `loss = α·KL(soft_student, soft_teacher) + (1-α)·CE(hard)` |

<a id="temperature-t"></a>
#### 软标签蒸馏的关键：温度 T

```python
T = 2.0   # 温度
teacher_probs = F.softmax(teacher_logits / T, dim=1)  # 软化分布
student_probs = F.log_softmax(student_logits / T, dim=1)
soft_loss = F.kl_div(student_probs, teacher_probs, log_target=True, reduction='batchmean') * (T * T)
```

> 🌰 **生活类比**：
> - 硬标签 = 老师只告诉学生答案"这道题选 B"
> - 软标签 = 老师告诉学生"B 是 70% 对，C 是 20%，A 和 D 各 5%"——信息量更大
> - 温度 T 越高，分布越平滑（软），知识传递越多
> - T=1 就是普通 softmax；T→∞ 就是均匀分布

#### 硬标签蒸馏代码（看 [hard_label_distillation.py](文本分类项目/06-model-compression/bert_distll/hard_label_distillation.py)）

```python
with torch.no_grad():
    teacher_logits = teacher_model(input_ids, attention_mask)
    teacher_preds = torch.argmax(teacher_logits, dim=1)  # 硬标签

student_logits = student_model(input_ids, attention_mask)
loss = nn.CrossEntropyLoss()(student_logits, teacher_preds)
```

#### 软标签蒸馏代码（看 [soft_label_distillation.py](文本分类项目/06-model-compression/bert_distll/soft_label_distillation.py)）

```python
T = 2.0
alpha = 0.7  # 软标签权重

# 软标签损失：学生学教师的概率分布
teacher_probs = F.softmax(teacher_logits / T, dim=1)
student_log_probs = F.log_softmax(student_logits / T, dim=1)
soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

# 硬标签损失：学生学教师的最终决定
hard_loss = nn.CrossEntropyLoss()(student_logits, teacher_preds)

# 总损失 = 软+硬加权
loss = alpha * soft_loss + (1 - alpha) * hard_loss
```

<a id="bilstm"></a>
### 6.4 BiLSTM 学生模型亮点（看 [bilstm_classifier.py](文本分类项目/06-model-compression/bert_distll/bilstm_classifier.py)）

```python
# 关键技巧：用 BERT 的 vocab 但模型用 BiLSTM
self.embedding = nn.Embedding(config.tokenizer.vocab_size, config.embed_size)
self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

# forward 时屏蔽 [CLS]/[SEP]/[PAD] 位置
cls_sep_mask = (input_ids != 101) & (input_ids != 102)
valid_mask = attention_mask & cls_sep_mask
embed = self.embedding(input_ids) * valid_mask.unsqueeze(-1)

# 双向 LSTM + 最大池化提取特征
lstm_out, _ = self.lstm(embed)
hidden, _ = (lstm_out * valid_mask.unsqueeze(-1)).max(dim=1)  # max pooling
logits = self.fc(self.dropout(hidden))
```

> 🌰 **生活类比**：
> - 双向 LSTM = 阅读时同时正着读和倒着读，理解更全面
> - 最大池化 = 整段话里"最有信号"的那个词决定整体类别
> - 屏蔽 [CLS]/[SEP] = 写作文不要把"标题"和"作者署名"也算成正文内容

---

## 七、文本分类项目知识脉络总结

### 7.1 演进路径与对比

| 模型 | 大小 | 推理速度 | 准确率 | 训练数据需求 | 推荐场景 |
|------|------|---------|-------|------------|---------|
| **随机森林+TFIDF** | ~10MB | 极快 | 75% | 中 | 快速基线 |
| **FastText** | ~50MB | 极快 | 88% | 中 | CPU部署、海量推理 |
| **BERT 微调** | 400MB | 较慢 | 94% | 中 | 高精度需求 |
| **量化BERT** | 100MB | 中 | 93% | - | 上线部署 |
| **剪枝BERT** | 280MB | 中 | 93% | - | 上线部署 |
| **蒸馏BiLSTM** | 5MB | 极快 | 91% | - | 移动端 |
| **LLM API** | 0 (云端) | 慢 | 90% | 0 | 冷启动、低频场景 |

### 7.2 关键设计决策清单

| 问题 | 推荐做法 |
|------|---------|
| 没有标注数据 | LLM 零样本 |
| 标注数据 < 1k | LLM Few-shot 或 数据增强 + FastText |
| 标注数据 1k~10k | BERT 微调 |
| 标注数据 > 10k | BERT 微调（效果最佳） |
| 部署在手机/IoT | BERT → 蒸馏 → BiLSTM |
| 部署在服务器，要求低延迟 | BERT → 量化 |
| 严苛延迟要求（<10ms） | FastText |
| 业务要求可解释 | TF-IDF + 随机森林 |

### 7.3 上线流程（每个项目都一样）

```
数据预处理 → 模型训练 → 模型保存 → 评估 → 模型压缩 → API封装 → UI/集成 → 监控
```

每一步在本项目都有对应实现：
- 数据预处理 → [01-data](文本分类项目/01-data/), [utils.py](文本分类项目/04-bert/src/utils.py)
- 模型训练 → [train.py](文本分类项目/04-bert/src/train.py)
- 评估 → `model2dev` 函数（贯穿所有项目）
- 压缩 → [06-model-compression](文本分类项目/06-model-compression/)
- API → 各个 `api.py`
- UI → 各个 `app.py`

---

## 八、常用技巧速查

### 8.1 加载预训练 BERT
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese")
```

### 8.2 文本编码
```python
encoding = tokenizer.encode_plus(
    text,
    max_length=32,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
```

### 8.3 推理时关闭梯度
```python
model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
```

### 8.4 早停 (Early Stopping)
```python
patience = 3
epochs_no_improve = 0
if dev_f1 > best_f1:
    best_f1 = dev_f1
    torch.save(model.state_dict(), path)
    epochs_no_improve = 0
else:
    epochs_no_improve += 1
    if epochs_no_improve >= patience:
        break  # 早停
```

> 🌰 **生活类比**：减肥期间连续 3 周体重不降反升，那就停止当前方法，换个食谱。

### 8.5 设备无关代码
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)
```

---

<a id="interview-questions"></a>
## 九、面试高频问题精选

**Q: BERT 的 [CLS] 为什么能代表整句？**
A: 预训练阶段 NSP 任务就是用 [CLS] 位置做二分类，倒逼它学习全局信息；微调时同样在 [CLS] 接分类头，所以它会被训练成"语义聚合器"。

**Q: 为什么 Transformer 用 LayerNorm 而不是 BatchNorm？**
A: NLP 序列长度不定、batch 内样本差异大，BN 不稳定；LN 在每个样本内部归一化，与 batch 无关，更适合 NLP。

**Q: 蒸馏时温度 T 的作用？**
A: T 越大，softmax 输出越平滑，类间相对差异更明显（"狗和猫的相似度" > "狗和汽车的相似度"），学生能学到更细粒度的"暗知识"。损失乘 `T²` 是为了梯度幅度与无温度时一致。

**Q: 量化精度损失大怎么办？**
A: ① 改用 QAT（量化感知训练）；② 关键层（如最后一层 Linear）保持 fp32；③ 使用更小的量化粒度（per-channel 而非 per-tensor）。

**Q: FastText 为什么这么快？**
A: ① 无复杂网络，只有 embedding + 平均 + 全连接；② 层级 Softmax 把 O(N) 变成 O(logN)；③ 用 C++ 实现，多线程并行。

**Q: 大模型时代为啥还学传统方法？**
A: ① 业务上 90% 场景不需要 LLM，传统方法够用且便宜；② 面试和工程实践都要懂；③ 模型压缩、特征工程的思想是相通的。

---

# 🎯 终极知识总结：通读全文后再回头看

> 看到这一章说明你已经把整篇文档读完了。这一章的目的是**把所有零散知识点串成一个心智模型**——不再按章节，而是按"做事时大脑里应该想什么"组织。
>
> ⚠️ 本章节的"一/二/三…"是独立编号，与前文"🚀 文本分类项目实战"的"七/八/九"是并列的两个 H1 大章节，不是续编。

## 一、AI 的"三件套"心智模型

任何一个 AI 任务，本质都是这三件事：

```
                 ┌──────────────────────────────┐
   数据 ─────►  │  模型 = 一堆带 W 和 b 的函数   │  ─────► 预测
                 └──────────────┬───────────────┘
                                │
                          损失函数 = 衡量"错多少"
                                │
                                ▼
                    优化器 = 用梯度调整 W、b
                                │
                                ▼
                          下一轮预测更准
```

记住这张图，你就理解了从 KNN 到 GPT 所有模型的本质——**只是这三件套的具体形态不同**。

| 模型 | 数据 | 模型形态 | 损失 | 优化方式 |
|------|------|----------|------|----------|
| KNN | 特征向量 + 标签 | 没有参数（懒惰学习）| 无 | 不优化（直接查近邻）|
| 线性回归 | 特征 + 数值 | `y = wx + b` | MSE | 梯度下降 |
| 随机森林 | 特征 + 类别 | 多棵决策树 | 基尼/熵 | 树构建（非梯度）|
| FastText | 词袋 + 类别 | Embedding + 平均 + Linear | Cross Entropy | SGD |
| BERT 微调 | tokens + 类别 | Transformer + 分类头 | Cross Entropy | AdamW |
| LLM API | Prompt + 输出 | 数十亿参数 Decoder | （已预训练）| 不动它，靠 Prompt |

> 🌰 **生活类比**：所有模型都是"考试机器"——学生（模型）看题（数据）写答案（预测），老师（损失函数）批改打分（错多少），学生（优化器）反思下次怎么写得更好（更新参数）。区别只在"学生大脑结构有多复杂"。

## 二、每一层的"为什么"链条（NLP 视角）

```
为什么要词嵌入？      → 因为"狗"和"猫"的 ID 没有语义距离，向量才有
   │
为什么要位置编码？    → 因为 Transformer 不像 RNN 那样按顺序处理，需要显式告诉它"谁先谁后"
   │
为什么要 padding？    → 因为 batch 内句子长短不一，不补齐没法做矩阵运算
   │
为什么要 mask？       → 因为 padding 是假数据不能算入注意力；解码器还要防止偷看未来
   │
为什么要多头注意力？  → 让模型同时关注不同方面（语法、语义、共指等），相当于多个"视角"并行
   │
为什么要 LayerNorm？  → 让每层输入分布稳定，加速收敛；相比 BatchNorm 更适合变长序列
   │
为什么要残差连接？    → 解决深层网络梯度消失，允许信息"绕过"某些层直接传播
   │
为什么要前馈层？      → 注意力是线性的（加权求和），FFN 提供非线性变换能力
```

> 🌰 **生活类比**：Transformer 像一台精密钟表，每个齿轮（组件）都有它存在的理由——拆掉任何一个钟就停了。

## 三、训练 vs 推理的全景对比

| 维度 | 训练阶段 | 推理阶段 |
|------|---------|---------|
| 是否需要标签 | ✅ 需要 | ❌ 不需要 |
| 是否计算梯度 | ✅ `requires_grad=True` | ❌ `with torch.no_grad():` |
| 是否更新权重 | ✅ `optimizer.step()` | ❌ 权重冻结 |
| 解码器输入 | 完整目标序列（Teacher Forcing）| 一步一步自回归生成 |
| Dropout / BN | ✅ 启用 | ❌ 关闭（`model.eval()`）|
| 关键代码 | `loss.backward()` + `optimizer.step()` | `model.eval()` + `with torch.no_grad():` |
| 关心的指标 | loss 是否下降 | latency、throughput、QPS |

> 🌰 **生活类比**：训练像驾校学车（教练在副驾，错了就打方向），推理像独立上路（自己开，出事自负）。

## 四、模型从 0 到上线的完整工程流（项目实战路径）

```
① EDA 数据探索  ──────►  搞清楚类别分布、句子长度分布
   ▼
② 选模型基线   ──────►  先跑 RF/FastText 拿基础分数，作为参照
   ▼
③ 主力模型训练  ──────►  BERT 微调，目标击败基线
   ▼
④ 评估调优     ──────►  classification_report + 混淆矩阵分析失败案例
   ▼
⑤ 模型压缩     ──────►  量化/剪枝/蒸馏选一个或组合，保精度降体积
   ▼
⑥ 上线部署     ──────►  FastAPI/Flask + 前端 UI + 接入业务系统
   ▼
⑦ 持续监控     ──────►  日志、漂移检测、数据回流再训练
```

> 💡 **重要**：本仓库的[文本分类项目实战](README.md#text-classification-project)严格按这个路线走，从 01-data 到 06-model-compression 是一条贯穿到底的工程线。

## 五、十大易踩的坑（按顺序）

1. **`requires_grad=True` 误以为是更新权重** → 它只是允许算梯度，更新靠 `optimizer.step()`
2. **忘记 `optimizer.zero_grad()`** → 梯度累加导致爆炸
3. **`nn.CrossEntropyLoss` 输入自己先 softmax** → 训练 loss 不下降
4. **二分类标签传 float 给 CE Loss** → 报 "expected scalar type Long"
5. **`backward()` 对向量调用** → 报错，要 `.sum().backward()`
6. **多分类用 Accuracy 评估**（类别不平衡）→ 用 weighted-F1 或 classification_report
7. **`roc_auc_score` 第二参数传类别**（应该传概率）→ 数值错误
8. **推理时不调 `model.eval()`** → Dropout/BN 仍生效，结果飘
9. **BERT 微调时 lr 用 0.001**（应该用 1e-5 ~ 5e-5）→ 大模型微调 lr 必须小
10. **蒸馏忘乘 T²** → 软标签梯度尺度错位，学生学得乱

## 六、不同 task 的"必备零件清单"

**做文本分类**：
- 分词器 / Tokenizer
- Embedding（或预训练模型加载）
- 序列模型（LSTM 或 Transformer）
- `[CLS]` 池化（BERT）或最大池化（LSTM）
- 分类头（Linear → num_classes）
- `nn.CrossEntropyLoss`
- `AdamW` / `SGD`
- 评估：`classification_report`

**做翻译/生成**：
- 上面所有 +
- 解码器（Decoder with Cross-Attention）
- 因果掩码 + padding 掩码
- BOS/EOS 特殊标记
- Teacher Forcing 训练 / 自回归推理
- BLEU / ROUGE 评估

**做模型压缩**：
- 教师模型（已训练好的大模型）
- 学生模型（更小架构，如 BiLSTM）
- 蒸馏损失（KLDivLoss + CrossEntropyLoss）
- 温度参数 T
- 或：动态量化 `torch.quantization.quantize_dynamic`
- 或：剪枝 `torch.nn.utils.prune`

## 七、面试时被问到任何模型，回答的"万能模板"

```
1. 它解决什么问题？               （任务定义）
2. 它的输入输出是什么形状？        （数据流）
3. 核心组件有哪几个？              （结构）
4. 训练时损失怎么算？              （损失函数）
5. 优化时用什么优化器和 lr？       （超参）
6. 它和 X 比有什么优势/劣势？      （对比）
7. 在你的项目里你怎么用的？        （结合实战）
```

举例 - 被问 "讲一下 BERT"：

> 1. 解决：通用语言表示，下游可以做分类、问答、NER 等
> 2. 输入：input_ids/attention_mask/token_type_ids 三个 [batch, seq_len]，输出 [batch, seq_len, 768]
> 3. 组件：Embedding (Token+Position+Segment) + 12 层 Transformer Encoder + 任务头
> 4. 预训练损失：MLM（遮 15% 词预测）+ NSP（判断两句话是否相邻）
> 5. 微调用 AdamW，lr=2e-5
> 6. 比 LSTM 强在双向 + 自注意力可并行；比 GPT 强在双向上下文
> 7. 我在 04-bert 项目里用 BertClassifier 包了 [CLS] 输出 + Linear(768, 10)，AdamW + 3 epoch 跑到 91% 准确率

## 八、整篇文档"一句话精华"

| 主题 | 一句话精华 |
|------|----------|
| 机器学习 | 给数据找一个数学函数让它能预测 |
| 损失函数 | 衡量预测错了多少的"标尺" |
| 交叉熵 | 衡量"预测概率分布"和"真实分布"的差距，二/多分类通用 |
| 混淆矩阵 | TP/FP/FN/TN 的 2×2 表，是所有分类指标的源头 |
| Autograd | PyTorch 自动给你算梯度，前提是 `requires_grad=True` |
| 5 步训练模板 | 前向 → 损失 → 清零 → 反向 → 更新 |
| RNN/LSTM/GRU | 把序列按时间步处理，靠"记忆细胞"传递历史 |
| 注意力 | 用 Q 查询 K，得到权重去加权 V，本质是"加权平均" |
| Transformer | 全靠注意力的并行序列模型，编码器理解、解码器生成 |
| BERT | 双向 Transformer 编码器，预训练 + 微调两阶段 |
| 模型压缩 | 量化（位数）+剪枝（连接）+蒸馏（学生学教师） |
| LLM | 大到不用训练，靠 Prompt 工程驱动它做下游任务 |

## 九、写在最后

学 AI 不是"背会哪些公式"，而是建立**心智模型**：
- 看到任何模型，能问"它的三件套是什么"
- 看到任何损失，能问"它在惩罚什么"
- 看到任何架构，能问"每一层为什么存在"
- 看到任何代码，能反推"这一行属于训练还是推理"

> 🌰 **最后一个生活类比**：学完整本书像学会做菜——
> - 你认识所有食材（PyTorch API）
> - 知道刀工火候（Autograd、optimizer、训练循环）
> - 懂得菜系搭配（CNN/RNN/Transformer/BERT）
> - 能根据食客口味调整（指标选择、模型压缩）
> - 最终自己开餐厅（上线部署）

下次遇到新模型，不要慌——**它不过是同一道家常菜的另一种炒法**。


