# data_loader.py 与 process.py 逐行详解

> 本文件对 CasRel（Cascade Binary Tagging）联合实体关系抽取模型的数据处理管线进行逐行解读，涵盖 `data_loader.py` 和 `process.py` 两个核心文件。每一行都会解释"做了什么"以及"为什么这样做"。

---

## 目录

- [Part 1: data_loader.py 逐行讲解](#part-1-dataloaderpy-逐行讲解)
  - [文件头部 (行 1-18)](#文件头部-行-1-18)
  - [MyDataset 类 (行 22-37)](#mydataset-类-行-22-37)
  - [get_data 函数 (行 40-68)](#getdata-函数-行-40-68)
  - [主程序入口 (行 71-80)](#主程序入口-行-71-80)
- [Part 2: process.py 逐行讲解](#part-2-processpy-逐行讲解)
  - [文件头部 (行 1-28)](#文件头部-行-1-28)
  - [find_head_idx 函数 (行 32-37)](#findheadidx-函数-行-32-37)
  - [create_label 函数 (行 41-78)](#createlabel-函数-行-41-78)
  - [collate_fn 函数 (行 81-129)](#collatefn-函数-行-81-129)
  - [extract_sub 函数 (行 134-147)](#extractsub-函数-行-134-147)
  - [extract_obj_and_rel 函数 (行 150-170)](#extractobjandrel-函数-行-150-170)
  - [convert_score_to_zero_one 函数 (行 173-179)](#convertscoretozeroone-函数-行-173-179)
- [Part 3: 整体数据流总结](#part-3-整体数据流总结)

---

## Part 1: data_loader.py 逐行讲解

### 文件头部 (行 1-18)

```python
"""
需求：准备训练、验证和测试数据集的DataLoader对象
思路步骤：
    1. 导入必要的库和配置
    2. 定义自定义的数据集类MyDataset：
        2.1 在构造函数中，读取指定路径的数据文件并解析为JSON格式存储
        2.2 实现__len__方法，返回数据集的长度
        2.3 实现__getitem__方法，获取指定索引的数据项，返回文本和spo列表
    3. 定义get_data函数：
        3.1 实例化训练、验证和测试数据集的MyDataset对象
        3.2 分别实例化训练、验证和测试数据集的DataLoader对象，设置批量大小、打乱顺序、整理函数等参数
        3.3 返回训练、验证和测试数据集的DataLoader对象
"""
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 1-13 | 多行文档字符串 | 描述整个模块的需求和实现思路 | 作为模块的设计蓝图，方便后续开发者快速理解代码结构。这里的"需求"回答了"这个模块要解决什么问题"，"思路步骤"回答了"如何一步步实现" |

```python
# coding:utf-8
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 14 | `# coding:utf-8` | 声明源文件编码为 UTF-8 | 确保 Python 解释器能正确处理文件中的中文字符（包括注释和字符串内容）。在 Python 3 中 UTF-8 是默认编码，但显式声明可以兼容某些旧版本工具 |

```python
from torch.utils.data import DataLoader, Dataset
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 15 | `from torch.utils.data import DataLoader, Dataset` | 从 PyTorch 导入数据加载的两个核心类 | **`Dataset`**：PyTorch 数据集的抽象基类，自定义数据集必须继承它并实现 `__len__` 和 `__getitem__`。**`DataLoader`**：将 Dataset 包装成可迭代的批量数据加载器，支持多线程加载、批处理、打乱等功能 |

```python
from codes.utils.process import *
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 16 | `from codes.utils.process import *` | 通配符导入 process.py 中的所有公开函数 | 将 `process.py` 中定义的 `collate_fn`、`find_head_idx`、`create_label` 等函数一次性导入，供 DataLoader 使用。**使用通配符导入 `*` 在工程实践中不太推荐**（容易造成命名空间污染），但这里因为 process.py 中的函数都与数据处理紧密相关，所以采用了便捷写法 |

```python
conf = Config()
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 18 | `conf = Config()` | 实例化配置对象 | 创建一个全局配置单例，包含模型路径、批量大小、设备类型等关键参数。此处作为**模块级变量**，意味着在 `import data_loader` 时就会被执行，后续所有函数都可以直接使用 `conf` |

---

### MyDataset 类 (行 22-37)

```python
# 自定义Dataset
class MyDataset(Dataset):
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 22 | `class MyDataset(Dataset):` | 定义一个继承自 `torch.utils.data.Dataset` 的自定义数据集类 | PyTorch 要求所有自定义数据集必须继承 `Dataset` 并实现 `__len__` 和 `__getitem__` 两个魔术方法。继承后，DataLoader 才能自动调用这些方法进行批量加载 |

---

```python
    # 在构造函数中，读取指定路径的数据文件并解析为JSON格式存储
    def __init__(self, data_path):
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 24 | `def __init__(self, data_path):` | 构造函数，接收一个 `data_path` 参数 | `data_path` 是 JSON 格式数据文件的路径。构造函数在创建 `MyDataset(data_path)` 实例时被自动调用 |

```python
        super(MyDataset, self).__init__()
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 25 | `super(MyDataset, self).__init__()` | 调用父类 `Dataset` 的构造函数 | 这是 Python 继承的标准实践，确保父类的初始化逻辑被执行。虽然 `Dataset.__init__` 本身不做什么，但保留这行可以确保未来 PyTorch 版本中父类添加初始化逻辑时不会出现兼容性问题 |

```python
        self.dataset = [json.loads(line) for line in open(data_path, encoding='utf8')]
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 26 | `self.dataset = [json.loads(line) for line in open(data_path, encoding='utf8')]` | 读取 JSON Lines 文件，将每一行解析为 Python 字典并存入列表 | 使用列表推导式逐行读取文件，每行是一个完整的 JSON 对象（包含 `text` 和 `spo_list` 字段），`json.loads()` 将其转为 Python 字典。**注意**：这里没有使用 `with` 语句，文件关闭依赖 Python 的垃圾回收机制，在生产代码中建议使用 `with open(...)` |

---

```python
    # 实现__len__方法，返回数据集的长度
    def __len__(self):
        return len(self.dataset)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 30 | `return len(self.dataset)` | 返回数据集中的样本总数 | PyTorch 的 `DataLoader` 需要通过此方法获知数据总量，以计算每个 epoch 包含多少个 batch、何时开始新 epoch 等。如果不实现这个方法，DataLoader 会报错 |

---

```python
    # 实现__getitem__方法，获取指定索引的数据项，返回文本和spo列表
    def __getitem__(self, index):
        content = self.dataset[index]
        text = content['text']
        spo_list = content['spo_list']
        return text, spo_list
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 33 | `def __getitem__(self, index):` | 定义按索引获取数据的方法 | DataLoader 在迭代时通过 `index` 从 0 到 `len-1` 依次（或随机）调用此方法 |
| 34 | `content = self.dataset[index]` | 取出第 `index` 个样本的字典 | 例如 `{"text": "苹果公司发布了iPhone", "spo_list": [...]}` |
| 35 | `text = content['text']` | 提取文本字段 | 原始文本字符串，如 `"苹果公司发布了iPhone"` |
| 36 | `spo_list = content['spo_list']` | 提取三元组列表 | **S**ubject-**P**redicate-**O**bject 列表，每个元素形如 `{"subject":"苹果公司","predicate":"发布","object":"iPhone"}` |
| 37 | `return text, spo_list` | 返回文本和三元组的元组 | 这个元组会被传入 `collate_fn` 进行批量整理 |

> **SPO 是什么？**
> S = Subject（主实体/主语）、P = Predicate（谓词/关系）、O = Object（客实体/宾语）。知识图谱的核心就是 `<主体, 关系, 客体>` 这样的三元组。

---

### get_data 函数 (行 40-68)

```python
def get_data():
    # 实例化训练数据集Dataset对象
    train_data = MyDataset(conf.train_data_path)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 42 | `train_data = MyDataset(conf.train_data_path)` | 创建训练集的 Dataset 对象 | 从配置中的 `train_data_path` 读取 `train.json`，将所有训练样本加载到内存。因为整体数据量通常不大（关系抽取数据集一般几千到几万条），全部载入内存是完全可行的 |

```python
    # 实例化验证数据集Dataset对象
    dev_data = MyDataset(conf.dev_data_path)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 44 | `dev_data = MyDataset(conf.dev_data_path)` | 创建验证集的 Dataset 对象 | **dev** = development，即开发集/验证集。训练过程中用它来监控模型是否过拟合、选择最佳超参数 |

```python
    # 实例化测试数据集Dataset对象
    test_data = MyDataset(conf.test_data_path)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 47 | `test_data = MyDataset(conf.test_data_path)` | 创建测试集的 Dataset 对象 | 测试集在训练完全结束后使用，用于评估模型的最终泛化能力。**测试集在整个训练过程中不能被模型"看到"** |

---

```python
    # 实例化训练数据集Dataloader对象
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=conf.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  drop_last=True)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 51-55 | `DataLoader(...)` | 创建训练集的 DataLoader | 各参数详解如下： |

| 参数 | 值 | 意义 |
|------|------|------|
| `dataset` | `train_data` | 要加载的数据集 |
| `batch_size` | `conf.batch_size` (=8) | 每个 batch 包含 8 个样本。batch_size 越小，梯度更新越频繁但噪声越大；越大则梯度估计越准确但显存消耗更多 |
| `shuffle` | `True` | 每个 epoch 随机打乱数据顺序。**训练集必须打乱**，防止模型学到数据的排列顺序而不是真正的模式 |
| `collate_fn` | `collate_fn` | 自定义批处理函数（来自 `process.py`）。PyTorch 默认的 collate 只会简单堆叠，而 NLP 任务需要对变长序列进行 padding |
| `drop_last` | `True` | 丢弃最后一个不足 batch_size 的 batch。避免最后一个 batch 样本数不同导致的 batch 统计（如 BatchNorm）不稳定 |

```python
    # 实例化验证数据集Dataloader对象
    dev_dataloader = DataLoader(dataset=dev_data,
                                batch_size=conf.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                drop_last=True)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 57-61 | 验证集 DataLoader | 创建验证集的 DataLoader | 参数与训练集基本相同。**验证集 `shuffle=True` 的影响不大**（因为验证阶段不计算梯度，顺序不影响模型更新），但设为 True 也没有负面影响 |

```python
    # 实例化测试数据集Dataloader对象
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=conf.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 63-67 | 测试集 DataLoader | 创建测试集的 DataLoader | 测试阶段打乱与否无所谓，因为只需要前向传播，不涉及梯度计算 |

```python
    return train_dataloader, dev_dataloader, test_dataloader
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 68 | `return ...` | 返回三个 DataLoader 对象 | 调用方（通常是训练脚本 `train.py`）拿到这三个对象后，可以直接用 `for batch in train_dataloader` 进行训练循环 |

---

### 主程序入口 (行 71-80)

```python
if __name__ == '__main__':
    train_dataloader, dev_dataloader, test_dataloader = get_data()
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 71-72 | `if __name__ == '__main__':` + `get_data()` | 当该文件作为脚本直接运行时执行测试代码 | Python 的惯用法：文件作为模块被 `import` 时，`__name__` 是模块名，不会执行这块代码；只有 `python data_loader.py` 直接运行时才会执行。**这是一个简单的集成测试** |

```python
    for inputs, labels in train_dataloader:
        for input in inputs:
            print(f'{input} --> {inputs[input].shape}')
        for label in labels:
            print(f'{label} --> {labels[label].shape}')
        print("*" * 100)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 74-80 | 循环打印张量形状 | 遍历一个 batch，打印每个输入张量和标签张量的形状 | 验证数据管线是否正常工作。通过查看 shape，可以确认：1) 数据能正确加载 2) collate_fn 能正确整理 3) 各张量的维度是否符合模型预期。例如 `input_ids` 的 shape 应该是 `(batch_size, seq_len)`，`sub_heads` 的 shape 应该是 `(batch_size, seq_len)` |

---

## Part 2: process.py 逐行讲解

### 文件头部 (行 1-28)

```python
"""
需求：实现数据处理相关功能，包括查找实体索引位置、创建标签以及整理数据批次。
思路步骤：
1. 准备：导入必要的模块和配置信息。
2. 定义`find_head_idx`函数...
3. 定义`create_label`函数...
4. 定义`collate_fn`函数...
"""
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 1-20 | 多行文档字符串 | 描述 process.py 的整体设计思路 | 明确了这个模块解决的三个核心问题：**定位实体位置**、**生成训练标签**、**整理批次数据** |

```python
# coding:utf-8
from codes.config import *
import torch
from random import choice
from collections import defaultdict
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 21 | `from codes.config import *` | 导入配置模块的全部内容 | 获取 `Config` 类以及其中的 `BertTokenizer`、`Vocabulary` 等 |
| 22 | `import torch` | 导入 PyTorch | 用于创建和操作张量（`torch.zeros`、`torch.tensor`、`torch.stack` 等） |
| 23 | `from random import choice` | 导入 `choice` 函数 | 用于从列表中**随机选择一个元素**。在 CasRel 中，一个句子可能包含多个主实体（Subject），训练时每次随机选择一个来构建标签 |
| 24 | `from collections import defaultdict` | 导入 `defaultdict` | 一个特殊的字典，访问不存在的 key 时自动创建默认值（这里是空 list），避免手动检查 key 是否存在 |

```python
# 配置对象实例化
conf = Config()
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 28 | `conf = Config()` | 实例化全局配置对象 | 与 data_loader.py 第 18 行功能相同。**注意**：同一个 `Config()` 在 data_loader.py 和 process.py 中各实例化了一次，所以内存中有两个独立的对象。不过因为 `Config` 只是存储静态参数，这不会造成问题 |

---

### find_head_idx 函数 (行 32-37)

```python
# 获取实体的开始索引位置
def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 32 | `def find_head_idx(source, target):` | 定义函数，在 `source` 中查找 `target` 子串的起始位置 | `source` 是经过 tokenizer 编码后的 token ID 列表，`target` 是实体对应的 token ID 列表 |
| 33 | `target_len = len(target)` | 获取目标实体的 token 长度 | 提前计算长度，避免在循环中反复调用 `len()` |
| 34 | `for i in range(len(source)):` | 遍历 source 的每个位置 | 从索引 0 开始逐个位置检查，**这是一种简单但时间复杂度为 O(n×m) 的暴力匹配** |
| 35 | `if source[i: i + target_len] == target:` | 切片比较 | 取 source 中从 `i` 开始、长度为 `target_len` 的子序列，逐个 token 与 target 比较是否完全一致 |
| 36 | `return i` | 找到则返回起始索引 | 返回第一个匹配位置。如果有多个相同实体，只返回第一个（但在知识图谱标注数据中，同一实体的多次出现通常指向同一位置） |
| 37 | `return -1` | 未找到返回 -1 | -1 作为"未找到"的标志，调用方需要检查这个值来决定是否忽略该三元组 |

> **为什么需要这个函数？**
> 原始数据中是字符串（如 `"苹果公司"`），但模型的输入是 token ID。必须找到实体在 token 序列中的精确位置，才能为模型生成正确的标签（即告诉模型"这里有一个实体的开始/结束"）。

---

### create_label 函数 (行 41-78)

这是整个数据处理管线中**最核心、最复杂**的函数。它负责将一个样本的三元组数据转换为模型训练所需的张量标签。

```python
# 获取每个样本的：主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
def create_label(inner_triples, inner_input_ids, seq_len):
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 41 | `def create_label(inner_triples, inner_input_ids, seq_len):` | 定义函数签名 | 三个参数：`inner_triples` = 该样本的三元组列表；`inner_input_ids` = 该样本经 tokenizer 编码后的 token ID 列表；`seq_len` = 序列长度（已 padding 到 batch 中最长句子的长度） |

---

#### 初始化零张量 (行 42-45)

```python
    inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
    inner_obj_heads = torch.zeros((seq_len, conf.num_rel))
    inner_obj_tails = torch.zeros((seq_len, conf.num_rel))
    inner_sub_head2tail = torch.zeros(seq_len)  # 随机抽取一个实体，从开头一个词到末尾词的索引
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 42 | `inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)` | 创建两个长度为 `seq_len` 的零向量 | **`sub_heads`**：标记主实体的**开始位置**（哪个 token 是主实体开头则为 1）。**`sub_tails`**：标记主实体的**结束位置**（哪个 token 是主实体结尾则为 1）。例如句子"苹果公司发布了iPhone"，主实体"苹果公司"对应 token 索引 [1,2]，则 `sub_heads[1]=1`，`sub_tails[2]=1` |
| 43 | `inner_obj_heads = torch.zeros((seq_len, conf.num_rel))` | 创建形状为 `(seq_len, num_rel)` 的二维零矩阵 | 每一列对应一种关系类型。例如第 3 列第 7 行为 1，表示"在 token 7 处，关系类型 3 对应的客实体开始" |
| 44 | `inner_obj_tails = torch.zeros((seq_len, conf.num_rel))` | 同上，标记客实体结束位置 | 与 `obj_heads` 配对使用，共同标记客实体的起止范围 |
| 45 | `inner_sub_head2tail = torch.zeros(seq_len)` | 创建主实体 span 掩码向量 | 当某个主实体被选中后，该实体从开头到结尾的所有 token 位置都标记为 1。这个信息帮助模型知道"当前正在处理哪个主实体的 span" |

---

#### 防止零除保护 (行 46-50)

```python
    # 因为数据预处理代码还待优化,会有不存在关系三元组的情况，
    # 初始化一个主词的长度为1，即没有主词默认主词长度为1，
    # 防止零除报错,初始化任何非零数字都可以，没有主词分子是全零矩阵
    inner_sub_len = torch.tensor([1], dtype=torch.float)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 50 | `inner_sub_len = torch.tensor([1], dtype=torch.float)` | 初始化主实体长度为 1（默认值） | 这是一个防御性编程的典型案例。有些样本可能没有标注三元组（数据不完整或本身就是负例），如果 `sub_len=0`，后续模型中的除法操作（如计算平均池化）会导致 NaN。设为 1 配合全零矩阵（分子为 0），结果为 0，不会报错 |

---

#### 构建 s2ro_map (行 51-67)

```python
    # 主词到谓词的映射
    s2ro_map = defaultdict(list)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 52 | `s2ro_map = defaultdict(list)` | 创建"主实体 → 客实体+关系"的映射表 | **s2ro** = **S**ubject **to** **R**elation & **O**bject。字典结构：`{(sub_head_idx, sub_tail_idx): [(obj_head_idx, obj_tail_idx, rel_idx), ...]}`。使用 `defaultdict(list)` 意味着访问不存在的 key 时自动创建空列表 |

```python
    for inner_triple in inner_triples:
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 54 | `for inner_triple in inner_triples:` | 遍历当前样本的每个三元组 | 一个句子可能有多对三元组，如"苹果公司发布了iPhone，总部在加州"就包含两个三元组 |

```python
        inner_triple = (
            conf.tokenizer(inner_triple['subject'], add_special_tokens=False)['input_ids'],
            conf.rel_vocab.to_index(inner_triple['predicate']),
            conf.tokenizer(inner_triple['object'], add_special_tokens=False)['input_ids']
        )
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 56-60 | 三元组编码 | 将原始的三元组字符串转换为 token ID 和关系 ID | 三部分分别处理：**Subject** = 用 BERT tokenizer 编码成 token ID 列表（`add_special_tokens=False` 不加 [CLS] 和 [SEP]）；**Predicate** = 通过 `rel_vocab.to_index()` 映射为关系类别编号（0~17）；**Object** = 同样用 tokenizer 编码 |

```python
        sub_head_idx = find_head_idx(inner_input_ids, inner_triple[0])
        obj_head_idx = find_head_idx(inner_input_ids, inner_triple[2])
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 61-62 | 查找实体位置 | 在句子的 token 序列中定位主实体和客实体的起始索引 | 调用 `find_head_idx` 进行子序列匹配。这里体现了该函数的核心用途：**将字符级别的实体标注映射到 token 级别的标注** |

```python
        if sub_head_idx != -1 and obj_head_idx != -1:
            sub = (sub_head_idx, sub_head_idx + len(inner_triple[0]) - 1)
            # s2ro_map保存主语到谓语的映射
            s2ro_map[sub].append(
                (obj_head_idx, obj_head_idx + len(inner_triple[2]) - 1, inner_triple[1]))
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 63 | `if sub_head_idx != -1 and obj_head_idx != -1:` | 只在主客实体都找到时才处理 | 如果 tokenizer 的分词方式导致实体无法在原句中精确匹配（比如 BERT 把"iPhone"分成了"I"和"##Phone"），则该三元组被跳过。这是一种**容错处理** |
| 64 | `sub = (sub_head_idx, sub_head_idx + len(inner_triple[0]) - 1)` | 构建主实体 span | 例如 token [1,2,3] 共 3 个 token，start=1，end=3（1+3-1） |
| 66-67 | `s2ro_map[sub].append(...)` | 将客实体+关系添加到对应主实体下 | 数据结构示例：`{(3,5): [(7,8,0), (10,12,3)]}` 表示主实体在位置 [3,5]，有两个对应客实体分别在 [7,8]（关系类型 0）和 [10,12]（关系类型 3） |

---

#### 生成标签张量 (行 68-78)

```python
    if s2ro_map:
        for s in s2ro_map:
            inner_sub_heads[s[0]] = 1
            inner_sub_tails[s[1]] = 1
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 68 | `if s2ro_map:` | 检查是否有有效的三元组映射 | 如果没有任何有效的三元组（所有主/客实体都匹配失败），则跳过，返回全零标签 |
| 69-71 | 循环设置 head/tail | 将所有主实体的开始/结束位置标记为 1 | **对所有候选主实体都做标记**。但在后续步骤中，只会随机选一个主实体来构建完整的训练标签 |

```python
        sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 72 | `choice(list(s2ro_map.keys()))` | 从所有主实体中**随机选择一个** | 这一行体现了 **CasRel 的核心训练策略**：每个训练样本一次只学习一个主实体及其对应的客实体和关系。如果一个句子有多个主实体，就随机选择一个。多轮训练下来，每个主实体都有机会被学习到。这样做的好处是**将复杂的多标签问题分解为每次只做一个二分类任务** |

```python
        inner_sub_head2tail[sub_head_idx:sub_tail_idx + 1] = 1
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 73 | `inner_sub_head2tail[sub_head_idx:sub_tail_idx + 1] = 1` | 将被选中主实体的整个 span 标记为 1 | 例如主实体在 [2,4]，则位置 2、3、4 都设为 1。这个向量的作用是：**告诉模型"当前正在关注哪个主实体"**，模型需要根据这个信息去预测对应的客实体和关系 |

```python
        inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 74 | `inner_sub_len = ...` | 记录被选中主实体的 token 长度 | 用于后续模型中的池化操作（对主实体 span 取平均时需要除以这个长度） |

```python
        for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
            inner_obj_heads[ro[0]][ro[2]] = 1
            inner_obj_tails[ro[1]][ro[2]] = 1
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 75-77 | 标记客实体位置 | 为被选中主实体对应的所有客实体设置标签 | `ro[0]` = 客实体起始位置，`ro[1]` = 客实体结束位置，`ro[2]` = 关系类型索引。在二维矩阵 `(seq_len, num_rel)` 中，**行 = token 位置，列 = 关系类型**。例如 `obj_heads[7][0]=1` 表示"关系类型 0 的客实体从 token 7 开始" |

```python
    return inner_sub_len, inner_sub_head2tail, inner_sub_heads, inner_sub_tails, inner_obj_heads, inner_obj_tails
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 78 | `return ...` | 返回 6 个张量 | 这 6 个张量完整描述了训练所需的全部标签信息 |

---

### collate_fn 函数 (行 81-129)

这个函数是 PyTorch DataLoader 的 `collate_fn` 参数，负责将一个 batch 的原始数据整理成模型可接受的张量格式。

```python
def collate_fn(data):
    text_list = [value[0] for value in data]
    triple = [value[1] for value in data]
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 81 | `def collate_fn(data):` | 定义 collate 函数 | `data` 是一个列表，每个元素是 `MyDataset.__getitem__` 返回的 `(text, spo_list)` 元组。列表长度 = batch_size |
| 82 | `text_list = [value[0] for value in data]` | 提取批次中所有文本 | 用列表推导式收集所有 `text`，例如 `["苹果公司发布了iPhone", "北京是中国的首都", ...]` |
| 83 | `triple = [value[1] for value in data]` | 提取批次中所有三元组 | 收集所有 `spo_list`，每个元素是原始三元组字典的列表 |

```python
    # 按照batch中最长句子补齐
    text = conf.tokenizer.batch_encode_plus(text_list, padding=True)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 85 | `text = conf.tokenizer.batch_encode_plus(text_list, padding=True)` | 批量 tokenize 并 padding | **`batch_encode_plus`** 是 HuggingFace tokenizer 的批量编码方法，一次性处理整个 batch。`padding=True` 表示将所有句子 pad 到 batch 内最长句子的长度，短句末尾补 `[PAD]` token（ID=0）。返回字典包含 `input_ids` 和 `attention_mask` |

```python
    batch_size = len(text['input_ids'])
    seq_len = len(text['input_ids'][0])
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 86-87 | 获取维度信息 | 确定当前 batch 的样本数和序列长度 | `batch_size`：当前 batch 实际样本数。`seq_len`：padding 后的统一序列长度（等于 batch 中最长句子的 token 数） |

```python
    sub_heads = []
    sub_tails = []
    obj_heads = []
    obj_tails = []
    sub_len = []
    sub_head2tail = []
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 88-93 | 初始化空列表 | 为 6 种标签类型各准备一个空列表 | 每个样本调用 `create_label` 后的返回结果会分别 append 到对应列表中，最后用 `torch.stack` 合并成 batch 维度的张量 |

```python
    # 循环遍历每个样本，将实体信息进行张量的转化
    for batch_index in range(batch_size):
        inner_input_ids = text['input_ids'][batch_index]  # 单个句子变成索引后
        inner_triples = triple[batch_index]
        # 获取每个样本的：主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
        results = create_label(inner_triples, inner_input_ids, seq_len)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 95-99 | 逐个样本生成标签 | 遍历 batch 中每个样本，调用 `create_label` | `inner_input_ids` 是 padding 后的完整 token ID 序列（包含 `[CLS]`、`[SEP]`、`[PAD]`）。**注意**：虽然句子被 pad 了，但 `create_label` 中的实体位置查找只匹配有效的 token 部分，padding 部分不会被标记 |

```python
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 100-105 | 收集标签 | 将 6 种标签分别存入对应列表 | `results[0]` 到 `results[5]` 对应 `create_label` 的 6 个返回值。这一步将"每个样本的标签"整理为"每种标签的所有样本" |

```python
    input_ids = torch.tensor(text['input_ids']).to(conf.device)
    mask = torch.tensor(text['attention_mask']).to(conf.device)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 106-107 | 转换并移动到设备 | 将 `input_ids` 和 `attention_mask` 转为张量并移到 GPU/CPU | `.to(conf.device)` 确保数据和模型在同一设备上。`attention_mask` 告诉模型哪些位置是真实 token（1）、哪些是 padding（0） |

```python
    # 借助torch.stack()函数沿一个新维度对输入batch_size张量序列进行连接，序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度,
    sub_heads = torch.stack(sub_heads).to(conf.device)
    sub_tails = torch.stack(sub_tails).to(conf.device)
    sub_len = torch.stack(sub_len).to(conf.device)
    sub_head2tail = torch.stack(sub_head2tail).to(conf.device)
    obj_heads = torch.stack(obj_heads).to(conf.device)
    obj_tails = torch.stack(obj_tails).to(conf.device)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 109-114 | `torch.stack` 堆叠 | 将每个样本的标签堆叠成 batch 维度的张量 | `torch.stack` 与 `torch.cat` 的区别：`stack` 会创建一个新维度（第 0 维是 batch），而 `cat` 在已有维度上拼接。最终张量形状示例：`sub_heads` → `(batch_size, seq_len)`，`obj_heads` → `(batch_size, seq_len, num_rel)` |

```python
    inputs = {
        'input_ids': input_ids,
        'mask': mask,
        'sub_head2tail': sub_head2tail,
        'sub_len': sub_len
    }
    labels = {
        'sub_heads': sub_heads,
        'sub_tails': sub_tails,
        'obj_heads': obj_heads,
        'obj_tails': obj_tails
    }
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 116-127 | 组织返回数据 | 将整理好的张量分为 `inputs` 和 `labels` 两个字典 | **清晰地区分了模型输入和训练标签**：`inputs` 包含 token、mask、主实体 span 提示和长度；`labels` 包含模型需要预测的目标（主实体位置、客实体位置） |

```python
    return inputs, labels
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 129 | `return inputs, labels` | 返回一个 batch 的数据 | 训练循环中可以用 `for inputs, labels in train_dataloader` 直接解包 |

---

### extract_sub 函数 (行 134-147)

这个函数在**推理/测试阶段**使用，用于从模型输出的概率向量中解码出预测的实体 span。

```python
def extract_sub(pred_sub_heads, pred_sub_tails):
    '''
    :param pred_sub_heads: 模型预测出的主实体开头位置
    :param pred_sub_tails: 模型预测出的主实体尾部位置
    :return: subs列表里面对应的所有实体【head, tail】
    '''
    subs = []
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 134-141 | 函数签名和文档字符串 | 定义提取主实体 span 的函数 | `pred_sub_heads` 和 `pred_sub_tails` 是模型输出的概率向量（形状均为 `(seq_len,)`），经 `convert_score_to_zero_one` 二值化后传入 |

```python
    # 统计预测出所有值为1的元素索引位置
    heads = torch.arange(0, len(pred_sub_heads), device=conf.device)[pred_sub_heads == 1]
    tails = torch.arange(0, len(pred_sub_tails), device=conf.device)[pred_sub_tails == 1]
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 142-143 | 布尔索引提取位置 | 找出所有值为 1 的元素在向量中的索引 | `torch.arange(0, n)` 生成 [0,1,2,...,n-1]，然后用布尔掩码 `[pred == 1]` 筛选出预测为"是实体边界"的位置。例如 `heads = [2, 10]` 表示模型预测位置 2 和 10 是实体开始 |

```python
    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 144-146 | 配对并过滤 | 将 heads 和 tails **一一配对**，过滤掉非法配对 | `zip(heads, tails)` 按顺序配对：第 1 个 head 配第 1 个 tail，第 2 个 head 配第 2 个 tail。**`tail >= head` 确保结束位置不在开始位置之前**，过滤掉无效的实体 span |

```python
    return subs
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 147 | `return subs` | 返回提取的实体列表 | 返回形如 `[(2, 4), (10, 12)]` 的列表，每个元素是一个 `(head, tail)` 元组 |

> **注意**：`zip` 的配对方式要求 heads 和 tails 长度相同且顺序一致。如果模型预测的 heads 和 tails 数量不同，多出的部分会被忽略。

---

### extract_obj_and_rel 函数 (行 150-170)

```python
def extract_obj_and_rel(obj_heads, obj_tails):
    '''
    :param obj_heads:  模型预测出的从实体开头位置以及关系类型
    :param obj_tails:  模型预测出的从实体尾部位置以及关系类型
    :return: obj_and_rels：元素形状：(rel_index, start_index, end_index)
    '''
    obj_heads = obj_heads.T
    obj_tails = obj_tails.T
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 158-159 | 转置 | 将 `(seq_len, num_rel)` 转置为 `(num_rel, seq_len)` | 转置后每一行对应一种关系类型，方便按关系遍历。原本 `obj_heads[i][r]` 表示"token i 是关系 r 的客实体开始"，转置后 `obj_heads[r][i]` 表示"关系 r 的客实体开始位置是否在 token i" |

```python
    rel_count = obj_heads.shape[0]
    obj_and_rels = []
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 160-161 | 初始化 | 获取关系类型数量（=18），初始化结果列表 | 需要遍历所有关系类型，对每种关系分别提取客实体 |

```python
    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]
        objs = extract_sub(obj_head, obj_tail)
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 162-165 | 逐关系提取 | 对每种关系类型，调用 `extract_sub` 提取该关系下的客实体 | 这里复用了 `extract_sub` 函数，体现了良好的代码复用。每种关系类型独立提取其对应的客实体 span |

```python
        if objs:
            for obj in objs:
                start_index, end_index = obj
                obj_and_rels.append((rel_index, start_index, end_index))
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 166-169 | 组装结果 | 将提取到的实体与关系索引绑定 | 最终格式 `(rel_index, start_index, end_index)`，例如 `(0, 7, 8)` 表示"关系类型 0 的客实体位于 token [7,8]" |

```python
    return obj_and_rels
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 170 | `return obj_and_rels` | 返回所有客实体-关系对 | 这些结果与 `extract_sub` 得到的主实体一起，可以还原为完整的三元组 `<sub, rel, obj>` |

---

### convert_score_to_zero_one 函数 (行 173-179)

```python
def convert_score_to_zero_one(tensor):
    '''
    以0.5为阈值，大于0.5的设置为1，小于0.5的设置为0
    '''
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor
```

| 行号 | 代码 | 做了什么 | 意义 |
|------|------|----------|------|
| 173-179 | 二值化函数 | 将模型输出的概率值（0~1）转换为 0/1 二值标签 | 模型最后一层使用 sigmoid 激活，输出 (0, 1) 之间的概率值。**0.5 作为默认阈值**：概率 ≥ 0.5 认为该位置有实体，否则没有。这是一个**就地修改**函数（in-place），直接修改输入 tensor，使用时需注意 |

---

## Part 3: 整体数据流总结

### 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据文件 (JSON Lines)                         │
│  {"text": "...", "spo_list": [{"subject":"..","predicate":"..",    │
│                                "object":".."}, ...]}                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  MyDataset (data_loader.py)                         │
│  • __init__: 逐行读取 JSON → 解析为 Python 字典列表                    │
│  • __len__:  返回数据集大小                                          │
│  • __getitem__: 返回 (text, spo_list) 元组                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  DataLoader (PyTorch)                               │
│  batch_size=8, shuffle=True, drop_last=True                        │
│  自动调用 collate_fn 整理每个 batch                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  collate_fn (process.py 行 81-129)                   │
│  1. 提取 texts 和 triples                                          │
│  2. batch_encode_plus → padding 到统一长度                           │
│  3. 逐样本调用 create_label → 生成标签张量                            │
│  4. torch.stack → 堆叠为 batch 维度                                 │
│  5. 移到 GPU/CPU，返回 (inputs, labels)                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  create_label (process.py 行 41-78)                  │
│  1. 初始化全零张量 (heads, tails, head2tail, len)                    │
│  2. tokenize 每个三元组的 S/P/O                                     │
│  3. find_head_idx 查找实体在句中的 token 位置                         │
│  4. 构建 s2ro_map: {(sub_span) → [(obj_span, rel)]}              │
│  5. 随机选一个主实体，设置对应的标签张量                                │
│  6. 返回 6 个标签张量                                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     模型输入 & 标签                                  │
│                                                                     │
│  inputs:                    labels:                                 │
│  ├─ input_ids (B, L)        ├─ sub_heads (B, L)                    │
│  ├─ mask (B, L)             ├─ sub_tails (B, L)                    │
│  ├─ sub_head2tail (B, L)    ├─ obj_heads (B, L, R)                 │
│  └─ sub_len (B, 1)          └─ obj_tails (B, L, R)                 │
│                                                                     │
│  (B=batch_size, L=seq_len, R=num_rel)                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 推理阶段的逆向流

```
模型输出 (概率张量)
     │
     ▼
convert_score_to_zero_one —— 以 0.5 为阈值二值化
     │
     ▼
extract_sub —— 从二值向量中提取 (head, tail) 实体 span
     │
     ▼
extract_obj_and_rel —— 按关系类型提取客实体，组装 (rel, start, end)
     │
     ▼
最终三元组 <sub, rel, obj>
```

### 关键设计决策

| 决策 | 解释 |
|------|------|
| **训练时随机选一个主实体** | CasRel 的级联策略：一次训练只关注一个主实体，将其编码为 `sub_head2tail`，然后预测该主实体对应的所有客实体和关系。多个 epoch 下来覆盖所有主实体 |
| **sub_len 默认为 1** | 防御性编程：防止无三元组样本导致的零除错误 |
| **find_head_idx 返回 -1** | tokenizer 分词可能导致实体无法精确匹配，用 -1 标志跳过该三元组，保证训练不中断 |
| **drop_last=True** | 避免最后一个不完整 batch 导致的 batch 统计方差问题 |
| **输入与标签分离** | `inputs` = 模型看到的，`labels` = 模型要预测的，清晰分离便于训练循环使用 |

### 两个文件的关系

```
data_loader.py ───import *───► process.py
      │                            │
      │  使用 collate_fn ◄─────────┘
      │  使用 create_label (间接，通过 collate_fn)
      │  使用 find_head_idx (间接，通过 create_label)
      │
      ▼
  返回 DataLoader 对象给训练脚本
```

- **`data_loader.py`** 是"门面"：提供对外接口（`get_data()` 函数），封装数据集的创建和 DataLoader 的配置
- **`process.py`** 是"引擎"：包含所有数据处理的核心逻辑（标签生成、批次整理、结果解码）
- 两者通过 `from codes.utils.process import *` 连接，`collate_fn` 是关键的桥梁
