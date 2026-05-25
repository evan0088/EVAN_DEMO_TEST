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
| KNN 分类算法 | [README.md:218](README.md#L218) | ML 基础 | "近朱者赤"——看周围 K 个邻居 |
| 数据特征处理 | [README.md:237](README.md#L237) | ML 基础 | 归一化/标准化/独热编码 |
| 损失函数（线性回归） | [README.md:263](README.md#L263) | ML 评估 | MAE/MSE/RMSE/R² |
| **交叉熵损失** | [README.md:276](README.md#L276) | ML 评估 | 二分类 & 多分类通吃 |
| **混淆矩阵** | [README.md:373](README.md#L373) | ML 评估 | TP/FP/FN/TN + P/R/F1/AUC |
| K-Means 聚类 | [README.md:505](README.md#L505) | ML 无监督 | 客户分群、肘部法则 |
| PyTorch 张量速查表 | [README.md:540](README.md#L540) | DL 基础 | 创建/形状/索引/运算 |
| **PyTorch 张量 18 例** | [README.md:570](README.md#L570) | DL 基础 | 创建→转换→运算→索引→形状→拼接 |
| **Autograd 自动微分** | [README.md:667](README.md#L667) | DL 基础 | 计算图 + `backward()` + `zero_grad()` |
| 5 步训练模板 | [README.md:755](README.md#L755) | DL 基础 | 前向→损失→清零→反向→更新 |
| 激活函数 | [README.md:905](README.md#L905) | DL 核心 | Sigmoid/ReLU/Tanh/Softmax |
| 文本预处理 | [README.md:928](README.md#L928) | NLP 基础 | 分词/去停用词/向量化 |
| RNN/LSTM/GRU | [README.md:948](README.md#L948) | NLP 序列模型 | 记忆细胞 + 门控机制 |
| 注意力机制四种 | [README.md:1067](README.md#L1067) | NLP 进阶 | 软/硬/加性/缩放点积 |
| **Transformer 完整架构** | [README.md:1327](README.md#L1327) | NLP 核心 | Encoder + Decoder + Mask |
| **编码器⇄解码器全链路** | [README.md:1558](README.md#L1558) | NLP 核心 | memory 纽带 + Q/K/V 来源 |
| 学习路线建议 | [README.md:1799](README.md#L1799) | 路线图 | ML→DL→NLP 三阶段 |
| 文本分类实战入口 | [README.md:1896](README.md#L1896) | 项目实战 | 6 阶段递进路线 |
| 01-data 数据 EDA | [README.md:1941](README.md#L1941) | 实战阶段 1 | 数据清洗 + padding 选择 |
| 02-rf 随机森林 | [README.md:1968](README.md#L1968) | 实战阶段 2 | TF-IDF + RF 基线 |
| 03-fasttext | [README.md:2009](README.md#L2009) | 实战阶段 3 | 字 vs 词级别 + autotune |
| **04-bert 微调** | [README.md:2071](README.md#L2071) | 实战阶段 4 | [CLS] + 三个 ID + 微调 |
| 05-LLM 大模型 | [README.md:2188](README.md#L2188) | 实战阶段 5 | DeepSeek API + Prompt 工程 |
| **06-model-compression** | [README.md:2258](README.md#L2258) | 实战阶段 6 | 量化/剪枝/蒸馏三板斧 |
| 面试高频题精选 | [README.md:2532](README.md#L2532) | 复习 | 6 道经典题 + 答案 |

### 四、学习路径推荐（三档进阶）

| 路径 | 周期 | 学完目标 | 推荐章节顺序 |
|------|------|---------|-------------|
| 🌱 **新手** | ~30 天 | 看懂代码、能跑模型 | 环境安装 → ML 基础 → PyTorch 张量 → Autograd → 5 步模板 → 激活函数 → RNN → 注意力 → Transformer 整体架构 |
| 🌿 **进阶** | ~60 天 | 能微调 BERT 完成业务分类 | 在新手基础上 + 编码器⇄解码器细节 → BERT 输入三件套 → 04-bert 全套代码 → 02-rf / 03-fasttext 对比 |
| 🌳 **专家** | ~90 天 | 模型上线 + 压缩部署 | 进阶基础上 + 蒸馏温度 T → 量化原理 → 剪枝 L1 → 上线四件套 → 05-LLM Prompt 工程 |

### 五、按需查找索引（"我想做 X" → "看 Y 章"）

| 我想… | 看哪章 |
|------|--------|
| 给老板做客户分群 | K-Means [README.md:505](README.md#L505) |
| 做一个垃圾邮件二分类器 | 混淆矩阵 [README.md:373](README.md#L373) + 02-rf [README.md:1968](README.md#L1968) |
| 做一个 10 类新闻分类器 | 04-bert [README.md:2071](README.md#L2071) |
| 模型太大想压缩 | 06-model-compression [README.md:2258](README.md#L2258) |
| 想理解 Transformer 怎么工作 | 编码器⇄解码器全链路 [README.md:1558](README.md#L1558) |
| 训练 loss 不下降怎么办 | 5 步训练模板 [README.md:755](README.md#L755) + Autograd 报错速查 [README.md:878](README.md#L878) |
| 想把模型上线提供 API | 上线四件套 [README.md:1997](README.md#L1997) |
| 不想训练，直接用 GPT 做分类 | 05-LLM [README.md:2188](README.md#L2188) |
| 评估指标怎么选 | 交叉熵 [README.md:276](README.md#L276) + 决策树 [README.md:473](README.md#L473) |
| 调 padding_size 时拍多少？ | EDA padding 选择 [README.md:1959](README.md#L1959) |
| 蒸馏温度 T 怎么设？ | 软标签蒸馏 [README.md:2365](README.md#L2365) |
| BERT 三个 ID 是啥？ | BERT 输入 [README.md:2090](README.md#L2090) |

### 六、关键术语索引（A-Z 速查）

| 术语 | 中文 | 行号 |
|------|------|------|
| **Attention** | 注意力机制 | [README.md:1067](README.md#L1067) |
| **Autograd** | 自动微分 | [README.md:667](README.md#L667) |
| **AUC / ROC** | ROC 曲线下面积 | [README.md:424](README.md#L424) |
| **BCE Loss** | 二元交叉熵 | [README.md:284](README.md#L284) |
| **BERT** | 双向编码 Transformer | [README.md:2071](README.md#L2071) |
| **BiLSTM** | 双向 LSTM | [README.md:2409](README.md#L2409) |
| **`backward()`** | 反向传播触发 | [README.md:700](README.md#L700) |
| **`[CLS]`** | 句子分类标记 | [README.md:2100](README.md#L2100) |
| **Cross-Attention** | 编码-解码交叉注意力 | [README.md:1558](README.md#L1558) |
| **Cross Entropy** | 交叉熵损失 | [README.md:276](README.md#L276) |
| **Decoder** | 解码器 | [README.md:1519](README.md#L1519) |
| **Distillation** | 知识蒸馏 | [README.md:2345](README.md#L2345) |
| **Encoder** | 编码器 | [README.md:1444](README.md#L1444) |
| **F1 Score** | F1 分数 | [README.md:392](README.md#L392) |
| **FastText** | 浅层快速分类 | [README.md:2009](README.md#L2009) |
| **GRU** | 门控循环单元 | [README.md:1016](README.md#L1016) |
| **K-Means** | K 均值聚类 | [README.md:505](README.md#L505) |
| **KNN** | K 近邻 | [README.md:218](README.md#L218) |
| **LayerNorm** | 层归一化 | [README.md:1444](README.md#L1444) |
| **LSTM** | 长短时记忆 | [README.md:981](README.md#L981) |
| **mask** | 掩码 | [README.md:1416](README.md#L1416) |
| **memory** | 编码器输出 | [README.md:1558](README.md#L1558) |
| **Multi-Head Attention** | 多头注意力 | [README.md:1444](README.md#L1444) |
| **NSP** | 下一句预测（BERT 任务）| [README.md:2081](README.md#L2081) |
| **Padding Size** | 序列填充长度 | [README.md:1959](README.md#L1959) |
| **Positional Encoding** | 位置编码 | [README.md:1368](README.md#L1368) |
| **Precision / Recall** | 精确率 / 召回率 | [README.md:392](README.md#L392) |
| **Pruning** | 剪枝 | [README.md:2305](README.md#L2305) |
| **Quantization** | 量化 | [README.md:2278](README.md#L2278) |
| **`requires_grad`** | 梯度跟踪标记 | [README.md:805](README.md#L805) |
| **RNN** | 循环神经网络 | [README.md:950](README.md#L950) |
| **Self-Attention** | 自注意力 | [README.md:1444](README.md#L1444) |
| **Softmax** | 多分类输出层 | [README.md:306](README.md#L306) |
| **Teacher Forcing** | 教师强制（训练）| [README.md:1558](README.md#L1558) |
| **TF-IDF** | 词频-逆文档频率 | [README.md:1976](README.md#L1976) |
| **Transformer** | Transformer 架构 | [README.md:1327](README.md#L1327) |
| **`zero_grad()`** | 梯度清零 | [README.md:712](README.md#L712) |

> 💡 **使用建议**：
> - **第一次学**：按"四"的新手路径走，**严禁跳级**
> - **复习**：先看"二"的依赖图回忆全貌，再跳到具体章节细读
> - **解决问题**：直接查"五"的反向索引或"六"的术语表
> - **面试前**：从顶到尾过一遍"三"的章节地图，重点章节加粗

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
│   ├── pytorch框架各种api示例/ # PyTorch基础
│   └── 神经网络/               # 神经网络基础
├── NLP/                        # 自然语言处理
├── pdf/                        # 学习资料
└── README.md                   # 项目说明
```

## 环境安装

```shell
pip install scikit-learn torch numpy pandas matplotlib jieba
```

## 人工智能发展的三要素

| 要素 | 说明 | 重要性 |
|------|------|--------|
| **数据** | 决定了模型最终效果的上限 | 🌟🌟🌟🌟🌟 |
| **算法** | 解决问题的思路/方法 | 🌟🌟🌟🌟 |
| **算力** | CPU/GPU/TPU等计算资源 | 🌟🌟🌟 |

> 💡 **核心理念**: 数据质量 > 数据数量 > 算法优化 > 算力提升

## 算法的学习方式

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

### 数据特征处理

数据特征处理是机器学习中至关重要的预处理步骤，主要包括**归一化**和**标准化**两种方法。

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

### (损失函数)交叉熵损失（Cross Entropy Loss）—— 二分类 & 多分类通吃

> ❗ **常见误解澄清**：交叉熵 **不只用于二分类**。它对二分类、多分类、多标签分类**都通用**，二分类只是它的一个特例。本仓库的 BERT 文本分类（10 类）、蒸馏学生模型，全部使用 `nn.CrossEntropyLoss`。

#### 一、核心思想（一句话）

**衡量"预测概率分布"和"真实分布"之间的差距**——预测越接近真实，损失越小；越偏离，损失越大。

#### 二、三种场景对照表

| 场景 | 标签形式 | 输出层 | 损失公式 | PyTorch API |
|------|---------|--------|----------|-------------|
| **二分类**（猫 vs 狗） | 0 或 1 | Sigmoid → 1 个概率 | `L = -[y·log(p) + (1-y)·log(1-p)]` | `nn.BCELoss` 或 `nn.BCEWithLogitsLoss`（推荐） |
| **多分类**（10 个新闻类别，互斥） | 0~9 之一 | Softmax → C 个概率 | `L = -Σᵢ yᵢ·log(pᵢ)`（yᵢ 是 one-hot） | **`nn.CrossEntropyLoss`**（最常用） |
| **多标签**（一篇文章可同时是"科技"+"财经"） | 多 hot 向量 | 每个类别独立 Sigmoid | 每类单独算 BCE，再求和/平均 | `nn.BCEWithLogitsLoss` |

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

#### 六、本仓库实战示例

| 文件 | 任务 | 损失函数 |
|------|------|----------|
| [文本分类项目/04-bert/](文本分类项目/04-bert/) | 10 类新闻分类（多分类） | `nn.CrossEntropyLoss()` |
| [06-model-compression/bert_distll/hard_label_distillation.py](文本分类项目/06-model-compression/bert_distll/hard_label_distillation.py) | 硬标签蒸馏 | `nn.CrossEntropyLoss()(student_logits, teacher_preds)` |
| [06-model-compression/bert_distll/soft_label_distillation.py](文本分类项目/06-model-compression/bert_distll/soft_label_distillation.py) | 软标签蒸馏 | `KLDivLoss + CrossEntropyLoss` 加权（见后文） |

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

### (损失函数)二分类模型评估指标 —— 混淆矩阵

#### 一、混淆矩阵长什么样（先看图）

| 预测\实际 | 正例 (Positive) | 负例 (Negative) |
|-----------|----------------|----------------|
| **正例**  | TP (真阳性)     | FP (假阳性/误报) |
| **负例**  | FN (假阴性/漏报) | TN (真阴性)     |

**4 个格子的助记口诀**：
- **第一个字母**：T/F = 预测**对没对**（True 对、False 错）
- **第二个字母**：P/N = 预测**说是正还是负**（Positive 正、Negative 负）

> 🌰 **生活类比 · 新冠核酸检测**：
> - **TP**：阳性病人 → 检出阳性 ✅（确实有病，也查出来了）
> - **FP**：健康人 → 检出阳性 ❌（误报，好人被冤枉）
> - **FN**：阳性病人 → 检出阴性 ❌（漏报，传染源放走了，最危险！）
> - **TN**：健康人 → 检出阴性 ✅（确实没病，也没冤枉）

#### 二、衍生指标（5 个最常用）

| 指标 | 公式 | 含义 | 适用场景 |
|------|------|------|----------|
| **准确率 (Accuracy)** | `(TP+TN)/(TP+TN+FP+FN)` | 整体预测正确的比例 | 类别平衡的数据集 |
| **精确率 (Precision)** | `TP/(TP+FP)` | **预测为正**的里头有多少真的是正 | 关注误报成本（如垃圾邮件） |
| **召回率 (Recall)** | `TP/(TP+FN)` | **真正是正**的里头有多少被找出来 | 关注漏报成本（如疾病检测） |
| **F1 分数** | `2·P·R/(P+R)` | 精确率和召回率的调和平均 | 需要兼顾两者 |
| **特异度 (Specificity)** | `TN/(TN+FP)` | 真正是负的里头有多少被识别 | 医学检测、ROC 曲线 |

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

> 💡 **重要提示**: 
> - 类别不平衡时，**不要只看准确率**！
> - 医疗诊断：优先保证高 **Recall**（宁可误报，不可漏报）
> - 垃圾邮件：优先保证高 **Precision**（宁可漏报，不可误报）
> - 不平衡多分类：用 **weighted-F1** 或 **macro-F1**，加 `classification_report` 看每类细节

### 无监督学习

| 算法 | 核心思想 | 适用场景 | 优缺点 |
|------|---------|---------|--------|
| **K-Means聚类** | 迭代更新簇中心，最小化簇内距离 | 客户分群、图像分割 | ✅ 简单高效 ❌ 需指定K值，对异常值敏感 |
| **层次聚类** | 构建树状结构，逐步合并或分裂簇 | 小数据集、生物信息学 | ✅ 无需指定K值 ❌ 计算复杂度高 |
| **DBSCAN** | 基于密度的聚类，识别任意形状簇 | 噪声数据、空间聚类 | ✅ 自动确定簇数，抗噪声 ❌ 参数敏感 |
| **PCA降维** | 线性变换，保留最大方差方向 | 数据可视化、特征压缩 | ✅ 去相关，降噪 ❌ 仅线性关系 |

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

## PyTorch基础

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

### PyTorch 张量 API 详细教程（18 个示例文件）

> 📂 文件来源：[Deep learnning/pytorch框架各种api示例/](Deep%20learnning/pytorch框架各种api示例/) 共 18 个 `.py` 文件，每个聚焦一个主题。下面按"创建 → 转换 → 运算 → 索引 → 形状 → 拼接"的脉络梳理。

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

#### 2️⃣ 数据类型与互转（[文件05](Deep%20learnning/pytorch框架各种api示例/05_张量的数据类型转换.py) ~ [文件07](Deep%20learnning/pytorch框架各种api示例/07_张量和标量互转.py)）

| 操作 | 示例 | 用途 |
|------|------|------|
| 看类型 | `tensor.dtype` | 查身份证 |
| 转浮点 | `tensor.float()` / `.type(torch.float32)` | 神经网络默认要 float |
| 转整型 | `tensor.int()` / `.long()` | 索引和标签必须 long |
| numpy → tensor | `torch.from_numpy(arr)` | **共享内存**（一改全改） |
| tensor → numpy | `tensor.numpy()` | 同样共享内存 |
| 单元素 → 标量 | `tensor.item()` | 损失值打印必备 |

> ⚠️ **坑点**：`torch.from_numpy()` 和 `.numpy()` 都是"借东西"不是"复制"，原地修改会互相污染。要彻底分开用 `.numpy().copy()`。

#### 3️⃣ 数学运算与聚合（[文件08](Deep%20learnning/pytorch框架各种api示例/08_张量的加减乘除负号基本运算.py) ~ [文件10](Deep%20learnning/pytorch框架各种api示例/10_张量的其他运算函数.py)）

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

#### 4️⃣ 索引切片（[文件11](Deep%20learnning/pytorch框架各种api示例/11_张量的基础索引操作.py) ~ [文件12](Deep%20learnning/pytorch框架各种api示例/12_张量的多维索引.py)）

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

#### 5️⃣ 形状变换（[文件13](Deep%20learnning/pytorch框架各种api示例/13_张量获取形状和修改形状.py) ~ [文件17](Deep%20learnning/pytorch框架各种api示例/17_张量的是否连续判断以及修改操作.py)）

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

#### 6️⃣ 拼接合并（[文件18](Deep%20learnning/pytorch框架各种api示例/18_张量的拼接操作.py)）

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

## 自动微分（Autograd）从零理解

> 📂 文件来源：[Deep learnning/神经网络/自动微分/](Deep%20learnning/神经网络/自动微分/)

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

### 三步标准模板（[01_单轮.py](Deep%20learnning/神经网络/自动微分/01_自动微分_更新权重_单轮.py)）

```python
# 1. 启用梯度跟踪
w = torch.tensor(10.0, requires_grad=True)

# 2. 正向计算 + 计算损失
loss = (w - 5) ** 2

# 3. 反向传播，计算梯度
loss.backward()
print(w.grad)  # → tensor(10.) 表示 dloss/dw = 10
```

### `backward()` 必须对标量调用

```python
y = x ** 2          # y 是向量
y.backward()        # ❌ 报错！必须是标量
y.sum().backward()  # ✅ 标量
```

> 🌰 **生活类比**：
> - 反向传播是"算总分对每道题的依赖"，必须有一个"总分"才能反推。
> - 向量没有"总分"概念，所以要 `.sum()` 加起来变成标量。

### 多轮训练的"梯度清零"陷阱（[02_多轮.py](Deep%20learnning/神经网络/自动微分/02_自动微分_更新权重_多轮.py)）

```python
for epoch in range(100):
    loss = (w - 5) ** 2
    loss.backward()           # ⚠️ 梯度会累加！
    
    with torch.no_grad():
        w -= 0.1 * w.grad     # 手动更新
    
    w.grad.zero_()            # 必须清零，否则下轮梯度叠加上来
```

> 🌰 **生活类比**：体重秤每天用前要清零，不然今天 70 kg 第二天就显示 140 kg；梯度也一样，不清零会越积越大，模型直接爆炸。

### 全连接 z = x @ w + b 的求导（[03_全连接.py](Deep%20learnning/神经网络/自动微分/03_自动微分_整体应用_推导wb梯度.py)）

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







# 深度学习



## 四大激活函数

| 激活函数 | 公式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|------|----------|
| **Sigmoid** | `σ(x) = 1/(1+e^(-x))` | 输出范围(0,1)，适合概率输出 | 梯度消失，输出非零中心 | 二分类输出层 |
| **Tanh** | `tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))` | 零中心化，收敛更快 | 梯度消失 | RNN隐藏层 |
| **ReLU** | `max(0, x)` | 计算简单，缓解梯度消失 | Dead ReLU问题（负区间梯度为0） | 大多数隐藏层（默认首选） |
| **Leaky ReLU** | `max(αx, x), α≈0.01` | 解决Dead ReLU | 效果不稳定，α需调参 | ReLU效果不佳时尝试 |

#### 激活函数选择建议

| 网络层次 | 推荐激活函数 | 原因 |
|---------|------------|------|
| **输入层** | 无需激活函数 | 直接传入原始特征 |
| **隐藏层** | ReLU / Leaky ReLU | 计算高效，缓解梯度消失 |
| **输出层-二分类** | Sigmoid | 输出概率值 [0,1] |
| **输出层-多分类** | Softmax | 输出概率分布 |
| **输出层-回归** | 无需激活函数 / Linear | 直接输出连续值 |
| **RNN/LSTM** | Tanh | 零中心化，稳定梯度 |


## NLP

### 文本预处理流程

| 步骤 | 方法 | 工具/库 | 说明 |
|------|------|---------|------|
| **1. 分词** | jieba分词、空格分割 | `jieba`、`split()` | 将句子切分为词语 |
| **2. 去停用词** | 过滤常见无意义词 | 停用词表 | 去除“的”、“是”等 |
| **3. 词性标注** | POS Tagging | `jieba.posseg` | 标注名词、动词等 |
| **4. 命名实体识别** | NER | spaCy、NLTK | 识别人名、地名、组织名 |
| **5. 向量化** | One-Hot、Word2Vec、Embedding | gensim、torch.nn.Embedding | 将词转为向量 |

### 词表示方法对比

| 方法 | 维度 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **One-Hot** | 词汇表大小 | 简单直观 | 稀疏、无语义信息 | 小规模数据集 |
| **TF-IDF** | 词汇表大小 | 考虑词频和文档频率 | 仍为稀疏向量 | 文本分类、检索 |
| **Word2Vec** | 自定义(50-300) | 捕捉语义关系 | 静态嵌入，一词一义 | 词相似度、类比任务 |
| **GloVe** | 自定义(50-300) | 全局统计信息 | 静态嵌入 | 通用词嵌入 |
| **BERT Embedding** | 768/1024 | 上下文相关，动态 | 计算成本高 | 现代NLP任务 |

### RNN 循环神经网络家族

#### 1. RNN (Recurrent Neural Network)
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

#### 2. LSTM (Long Short-Term Memory)
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

#### 3. GRU (Gated Recurrent Unit)
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

#### RNN vs LSTM vs GRU 对比总结

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

### 注意力机制 (Attention Mechanism)

#### 注意力机制核心概念

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

#### 三种注意力机制对比

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

#### 1. 软性注意力 (Soft Attention)

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

#### 2. 硬性注意力 (Hard Attention)

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

#### 3. 加性注意力 (Additive Attention)

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

#### 4. 缩放点积注意力 (Scaled Dot-Product Attention)

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

#### 选择建议

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

### Transformer 完整架构详解（NLP/12~18 文件）

> 📂 文件来源：[NLP/13~18.py](NLP/) + [encoder.py](NLP/encoder.py) / [decoder.py](NLP/decoder.py) / [input.py](NLP/input.py)，配套案例 [12.1-英译法案例.py](NLP/12.1-英译法案例.py)。

Transformer 是 2017 年 Google 论文 *Attention Is All You Need* 提出的架构，**抛弃 RNN，纯靠注意力**，奠定了 BERT/GPT/ChatGPT 的基石。

#### 整体架构图（极简版）

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

#### 关键超参（Transformer-base）

| 参数 | 值 | 含义 |
|------|-----|------|
| `d_model` | 512 | 词向量/隐层维度 |
| `num_heads` | 8 | 多头数 |
| `d_k` | 64 | 每头维度 = `d_model / num_heads` |
| `d_ff` | 2048 | 前馈网络隐层 |
| `N` | 6 | 编码器/解码器层数 |
| `max_len` | 60~512 | 最大序列长度 |

---

#### 1️⃣ 输入层（[13-input编码器之位置编码.py](NLP/13-input编码器之位置编码.py)）

##### 词嵌入层

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

##### 位置编码（核心公式）

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

##### 输入 = embedding + 位置编码

```python
x = embedding(x) * sqrt(d_model)
x = x + positional_encoding[:, :x.size(1)]
x = dropout(x)
```

---

#### 2️⃣ 掩码（[14-input编码器之mask掩码.py](NLP/14-input编码器之mask掩码.py)）

两种 mask 都是 0/1 矩阵，**0 的位置会被替换为 `-inf`**，softmax 后变 0。

##### Padding Mask（编码器+解码器都用）****

屏蔽 `[PAD]` 占位符，避免模型把"无意义填充"当成有效信息。
```python
padding_mask = (input_ids != 0).unsqueeze(-2)  # [batch, 1, seq_len]
```

> 🌰 **生活类比**：考试卷子上有些空白格不算分，阅卷老师直接跳过。

##### Causal Mask / Subsequent Mask（仅解码器）

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

#### 3️⃣ 编码器（[15-transform之encoder.py](NLP/15-transform之encoder.py) + [16-层标准化.py](NLP/16-transform之层标准化.py)）

##### 缩放点积注意力（一次计算）

```python
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 缩放
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = softmax(scores, dim=-1)
    return weights @ V, weights
```

##### 多头注意力（[15](NLP/15-transform之encoder.py)）

```python
# 把 d_model=512 拆成 8 头，每头 d_k=64
Q = self.W_q(x).view(batch, -1, num_heads, d_k).transpose(1, 2)
K = self.W_k(x).view(batch, -1, num_heads, d_k).transpose(1, 2)
V = self.W_v(x).view(batch, -1, num_heads, d_k).transpose(1, 2)

out, _ = attention(Q, K, V, mask)
out = out.transpose(1, 2).contiguous().view(batch, -1, d_model)
out = self.W_o(out)  # 最终线性层
```

> 🌰 **生活类比**：一个评委（单头）容易主观；8 个评委（多头）各看不同角度（语法/语义/情感/位置/...）再综合，更全面。

##### 前馈网络

```python
# 两层全连接 + ReLU
FFN(x) = Linear_2( ReLU( Linear_1(x) ) )
# 维度: 512 → 2048 → 512
```

> 🌰 **生活类比**：先把信息"展开"到大房间（2048）方便整理，再"压缩"回原房间（512）。

##### Add & Norm（残差 + 层归一化）

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

##### 编码器层堆叠 6 次

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

#### 4️⃣ 解码器（[17-transform-decoder.py](NLP/17-transform-decoder.py)）

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

---

#### 5️⃣ 输出层（[18-transform之output.py](NLP/18-transform之output.py)）

```python
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

把 512 维隐藏向量映射回**词表大小**，再 log_softmax 得到每个词的概率对数（配合 NLLLoss 用）。

> 🌰 **生活类比**：从"我懂了什么意思"（隐藏向量）翻译成"该说哪个具体的词"（词表概率）。

---

#### 6️⃣ 编码器 ⇄ 解码器 全链路联系（重点）

前面 5 节把 6 个组件单独讲了一遍，这一节回答最关键的问题：**它们之间到底怎么连？数据流到底长什么样？**

##### 6.1 数据流总览（带形状追踪）

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

##### 6.2 关键纽带：memory（编码器输出）

| 维度 | 说明 |
|------|------|
| 名称 | `memory`（也叫 `encoder_output`） |
| 形状 | `[batch, src_len, d_model]` |
| 含义 | 源句子每个词的"上下文增强表示"（每个词都已融合全局信息） |
| 用途 | 传入**每个**解码器层的 cross-attention，作为 K 和 V |
| 寿命 | 编码器跑一次后，整个解码过程都不变 |

> 🌰 **生活类比**：memory 像翻译考试时摆在桌上的英文原稿。无论你写到第几个汉字，都能随时回头看这份原稿；原稿不会变，永远只算一次。

##### 6.3 解码器三个子层的 Q/K/V 来源差异（最容易混的地方）

这是 Transformer 最精妙也最容易记混的一点：

| 子层 | Q 来自 | K 来自 | V 来自 | 用 mask | 作用 |
|------|--------|--------|--------|---------|------|
| ① Masked 自注意力 | 解码器当前层输入 | 同 Q | 同 Q | tgt_mask（padding+因果） | 看自己已生成的词 |
| ② Cross-Attention | ① 的输出 | **memory** | **memory** | src_mask（仅 padding） | 对齐到原文相关位置 |
| ③ 前馈 FFN | ② 的输出 | — | — | — | 非线性变换 |

代码对应（[17-transform-decoder.py:39-43](NLP/17-transform-decoder.py#L39-L43)）：
```python
# x = 解码器输入(目标词向量+位置编码), m = memory(编码器输出)
x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))   # ① Q=K=V=x
x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))    # ② Q=x, K=V=m
x = self.sublayer[2](x, self.feed_forward)                                 # ③ FFN
```

> 🌰 **生活类比**（译者翻英文）：
> - 子层① = 看自己刚写的几个汉字，保持上下文连贯
> - 子层② = 抬头看英文原稿，找下一个该翻什么（**Q 是脑中疑问，K/V 是原稿**）
> - 子层③ = 大脑加工一下，准备下笔

##### 6.4 为什么 cross-attention 中 K 和 V 都来自 memory？

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

##### 6.5 两种 mask 在编码器/解码器的分工

| Mask | 形状 | 编码器自注意力 | 解码器自注意力 | 解码器跨注意力 |
|------|------|--------------|---------------|---------------|
| **src_mask**（源 padding） | `[batch, 1, 1, src_len]` | ✅ 屏蔽源 PAD | ❌ | ✅ 屏蔽源 PAD |
| **tgt_mask**（目标 padding ∧ 因果） | `[batch, 1, tgt_len, tgt_len]` | ❌ | ✅ 屏蔽 PAD + 不看未来 | ❌ |

代码（[17-transform-decoder.py:78-86](NLP/17-transform-decoder.py#L78-L86)）：
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

##### 6.6 训练 vs 推理：数据流的根本差异

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

##### 6.7 编码器内部一层的形状追踪（多头注意力示例）

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

##### 6.8 完整 Transformer 一图记忆

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

##### 6.9 高频面试问答（Transformer 编解码器篇）

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

#### 7️⃣ 英译法 Seq2Seq + 加性注意力实战（[12.1](NLP/12.1-英译法案例.py) / [12.2](NLP/12.2-英译法案例.py)）

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

##### Transformer vs Seq2Seq+Attention 对比

| 维度 | Seq2Seq+Attn | Transformer |
|------|-------------|------------|
| 主体 | RNN/GRU/LSTM | 全注意力 |
| 并行 | ❌ 必须串行 | ✅ 全部并行 |
| 长序列 | 易遗忘 | 更稳健 |
| 训练速度 | 慢 | 快 5~10 倍 |
| 当前主流 | 已被淘汰 | BERT/GPT/Claude 的祖宗 |

---

#### Transformer 一句话记忆

> **"输入加位置，多头来注意，残差防梯消，前馈再过一遍，编完给解码，掩码盖未来，最后线性 + softmax 出词。"**

---



---

## 学习路线建议

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
2. ✅ 词表示方法（One-Hot、Word2Vec、Embedding）
3. ✅ RNN家族（RNN、LSTM、GRU）的原理和实现
4. ✅ 注意力机制（加性注意力、乘性注意力、缩放点积注意力）
5. ✅ 实战项目（英译法机器翻译）

---

## 常用API速查

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

### 1.3 padding_size 怎么选

不能选最大长度（浪费算力），不能选平均值（一半文本被截）。
**推荐选 95% ~ 99% 分位数**，让 95% 以上的文本不被截断，同时保持高效。

> 🌰 **生活类比**：买衣服尺码不会按部门里最高的人买（浪费布），也不会按平均身高（一半人穿不下），而是按 P95（95% 的人能穿）。

---

## 二、02-rf：随机森林 + TF-IDF（传统ML基线）

### 2.1 随机森林是什么

**随机森林 (Random Forest)** = 多棵决策树投票。

> 🌰 **生活类比**：让 100 个朋友帮你判断一条新闻是什么类别，每个朋友都给一个答案，最后少数服从多数。每个朋友（决策树）只看到部分特征（随机），所以判断角度不同，避免"群体盲点"。

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

### 2.4 上线四件套：训练 → 保存 → API → UI

这是工业界通用流程，每个项目都会重复：
1. **训练**：`fit()` 拟合数据
2. **保存**：`joblib.dump(model, "model.pkl")` 存盘
3. **API**：FastAPI 暴露 `/predict` 接口，输入文本返回类别
4. **UI**：网页让产品经理 / 用户自己测

> 🌰 **生活类比**：训练 = 学做菜；保存 = 把菜谱写下来；API = 在外卖平台开店；UI = 餐厅前台让顾客点菜。

---

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

## 四、04-bert：BERT 微调（重点章节）

### 4.1 BERT 是什么

**BERT (Bidirectional Encoder Representations from Transformers)**：Google 2018 年提出的预训练语言模型。

核心思想：**先在海量无标注语料上做"完形填空"预训练，再在你的小数据集上微调**。

> 🌰 **生活类比**：BERT 就像一个读完了"百度百科+维基百科+网络小说+新闻"的大学生。他不知道你公司的具体业务，但语文功底好，你只需要"实习培训"几天，他就能上岗。这比从初中生（随机初始化）开始训练快得多、效果好得多。

### 4.2 BERT 的两大预训练任务

| 任务 | 全称 | 做什么 | 学会了什么 |
|------|------|--------|----------|
| **MLM** | Masked Language Model | 随机遮住15%的词，让模型猜 | 词的双向上下文 |
| **NSP** | Next Sentence Prediction | 判断两句话是不是连续 | 句子级别的关系 |

> 🌰 **生活类比**：MLM = 完形填空；NSP = 判断"这两段话是不是同一篇文章里相邻的两段"。

### 4.3 BERT 输入的三个 ID

输入文本会被 tokenizer 转成三类 ID：

| ID 类型 | 维度 | 含义 |
|---------|------|------|
| `input_ids` | [batch, seq_len] | 每个 token 在词表中的编号 |
| `attention_mask` | [batch, seq_len] | 1=真实token，0=padding填充 |
| `token_type_ids` | [batch, seq_len] | 句子A=0，句子B=1（NSP用） |

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

> 💡 **重要**：本仓库的[文本分类项目实战](README.md#L1896)严格按这个路线走，从 01-data 到 06-model-compression 是一条贯穿到底的工程线。

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


