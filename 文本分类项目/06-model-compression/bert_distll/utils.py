# 文件名：utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import time
from datetime import timedelta
from config import Config
conf=Config()


def get_time_diff(start_time):
    """
    计算时间差，并根据持续时间动态格式化输出。

    参数：
        start_time (float): 开始时间（time.time() 返回的秒时间戳）。

    返回：
        str: 格式化的时间差字符串，单位根据时间长度选择：
             - 小于 1 秒：毫秒 (ms)
             - 1 秒到 1 分钟：秒和毫秒 (s)
             - 大于等于 1 分钟：分钟、秒和毫秒 (min s)
    """
    # 计算时间差（秒）
    end_time = time.time()
    diff_seconds = end_time - start_time

    # 转换为毫秒
    diff_milliseconds = diff_seconds * 1000

    if diff_seconds < 1:
        # 小于 1 秒，显示毫秒（保留 0 位小数）
        return f"{int(round(diff_milliseconds))} ms"
    elif diff_seconds < 60:
        # 1 秒到 1 分钟，显示秒和毫秒（保留 3 位小数）
        return f"{diff_seconds:.3f} s"
    else:
        # 大于等于 1 分钟，显示分钟、秒和毫秒
        minutes = int(diff_seconds // 60)
        seconds = diff_seconds % 60
        return f"{minutes} min {seconds:.3f} s"



def load_raw_data(file_path):
    """
    读取原始数据文件，解析为文本和标签。

    参数：
        file_path (str): 数据文件路径（如dev2.txt）。

    返回：
        List[Tuple[str, int]]: 包含(文本, 标签)的列表。
    """
    data = []
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f, desc="Loading data"):
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t")
            data.append((text, int(label)))
    return data


class TextDataset(Dataset):
    """
    自定义Dataset，存储原始文本和标签，用于BERT分类任务。
    """

    def __init__(self, data):
        """
        参数：
            data (List[Tuple[str, int]]): 原始数据，包含(文本, 标签)的列表。
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回单条样本的x（文本）和y（标签）。

        参数：
            idx (int): 样本索引。

        返回：
            Tuple[str, int]: 单条样本的(文本, 标签)，即(x, y)。
        """
        x, y = self.data[idx]
        return x, y


def collate_fn(batch):
    """
    DataLoader的collate_fn，处理分词、统一padding、mask生成和Tensor转换。

    参数：
        batch (List[Tuple[str, int]]): 批次数据，包含(文本, 标签)。
        tokenizer (BertTokenizer): BERT分词器。
        padding_size (int): 统一padding长度（默认28，基于文本长度统计）。
        device (str): 设备（"cpu"或"cuda"）。

    返回：
        Tuple[torch.Tensor, ...]: (input_ids, seq_len, attention_mask, labels) 的Tensor格式。
    """
    # 提取文本和标签
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 批量分词，自动添加 [CLS] 和 [SEP]
    text_tokens = conf.tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,  # 自动添加 [CLS] 和 [SEP]
        padding=True,  # 先不padding，稍后统一处理
        return_attention_mask=True,
    )
    token_ids_list = text_tokens["input_ids"]
    token_attention_mask_list = text_tokens["attention_mask"]
    # 转为 Tensor 并移到设备
    input_ids = torch.tensor(token_ids_list)
    attention_mask = torch.tensor(token_attention_mask_list)
    labels = torch.tensor(labels)
    #
    # print("================================")
    # print(labels)
    # print(attention_mask)
    # print(input_ids)




    return input_ids, attention_mask, labels


def build_dataloader():
    """
    构建DataLoader，整合数据加载、Dataset和collate_fn。

    参数：
        file_path (str): 数据文件路径。
        batch_size (int): 批次大小。
        padding_size (int): 统一padding长度（默认28）。
        device (str): 设备（"cpu"或"cuda"）。

    返回：
        DataLoader: 用于训练的DataLoader。
    """
    # 加载原始数据
    train_path=conf.train_path
    test_path=conf.test_path
    dev_path=conf.dev_path
    batch_size=conf.batch_size

    train_data = load_raw_data(train_path)
    test_data = load_raw_data(test_path)
    dev_data = load_raw_data(dev_path)

    # 创建 Dataset
    train_dataset = TextDataset(train_data)
    dev_dataset = TextDataset(dev_data)
    test_dataset = TextDataset(test_data)

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader,test_dataloader,dev_dataloader


# 示例用法
if __name__ == "__main__":
    # 配置参数
    file_path = r"/01-data/dev3.txt"
    batch_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 记录开始时间
    start_time = time.time()

    # 构建 DataLoader
    train_dataloader,test_dataloader,dev_dataloader = build_dataloader()

    # #遍历 DataLoader
    # for batch in train_dataloader:
    #     input_ids, attention_mask, labels = batch
    #     print("Input IDs:", input_ids.shape)
    #     print("Attention Mask:", attention_mask.shape)
    #     print("Labels:", labels.shape)
    #     breakpoint()