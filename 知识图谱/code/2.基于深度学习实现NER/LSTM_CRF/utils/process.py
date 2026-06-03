import json
import os

os.chdir('..')
cur = os.getcwd()
print('当前数据处理默认工作目录：', cur)

# os.path.dirname: 表示上级目录
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.abspath(os.path.dirname(path))

print(path)

"""
需求：将标注好的序列数据进行BIO标注
思路步骤：
1. 加载文件：需要加载json: ①labels ②tag2id ③序列数据目录 ④写入路径
2.基于深度学习实现NER. 循环读取每个原始序列文件: 原始文件 -> 对应的标注内容 文件对
  2.基于深度学习实现NER.1 通过for root, dirs, files in os.walk(self.origin_path)遍历序列数据目录下的每个文件
  2.基于深度学习实现NER.2.基于深度学习实现NER 对于每个文件进行文件类型过滤，只读取 "原始序列数据"
  2.基于深度学习实现NER.3 获取每个"原始序列数据"文件对应的"标注内容"文件
3. 获取每个序列的BIO标签
  3.1 基于原始序列根据标注信息，得到实体开始索引、结束索引、中文标签类型
  3.2.基于深度学习实现NER 基于labels.json得到中文标签类型到tag转换
  3.3 基于标注中的位置信息得到BIO标签
4. 标注数据并写入文件
	4.1 读取原始序列数据
	4.2.基于深度学习实现NER 结合BIO标签进行序列标注
	4.3 把标注好的数据写入文件
"""


class TransferData():
    # 1. 加载文件：需要加载json: ①labels ②tag2id ③序列数据目录 ④写入路径
    def __init__(self):
        # os.path.join ： cur + / + 'data/labels.json' TODO 这里不能再加反斜杠了
        self.label_dict = json.load(open(os.path.join(cur, 'data/labels.json')))
        self.seq_tag_dict = json.load(open(os.path.join(cur, 'data/tag2id.json')))
        self.origin_path = os.path.join(cur, 'data_origin')
        # 如果data目录是存在，使用python的写入的时候，它自动创建data下面的/train.txt。 如果目录不存在，报错
        self.train_filepath = os.path.join(cur, 'data/train.txt')


    def transfer(self):
        with open(self.train_filepath, 'w', encoding='utf-8') as fr:
            #  2.基于深度学习实现NER.1 通过for root, dirs, files in os.walk(self.origin_path)遍历序列数据目录下的每个文件
            # root: 当前遍历到的目录(根目录); dirs: 当前目录下有哪些文件夹(目录),list; files: 当前目录下有哪些文件,list
            for root, dirs, files in os.walk(self.origin_path):
                for file in files:
                    # TODO：原始序列数据：一般项目-1.txtoriginal.txt
                    filepath = os.path.join(root, file)
                    #  2.基于深度学习实现NER.2.基于深度学习实现NER 对于每个文件进行文件类型过滤，只读取 "原始序列数据"
                    if 'original' not in filepath:
                        continue
                    #  2.基于深度学习实现NER.3 获取每个"原始序列数据"文件对应的"标注内容"文件
                    # 删掉 .txtoriginal  ： 一般项目-1.txtoriginal.txt -> 一般项目-1.txt
                    # TODO：标注数据：一般项目-1.txt
                    label_filepath = filepath.replace('.txtoriginal','')
                    # print(filepath, '\t\t', label_filepath)
                    # 获取每个序列的BIO标签
                    res_dict = self.read_label_text(label_filepath)
                    with open(filepath, 'r', encoding='utf-8')as f:
                        # TODO strip去掉文本中的多余的空格
                        content = f.read().strip()
                        for indx, char in enumerate(content):
                            char_label = res_dict.get(indx, 'O')
                            fr.write(char + '\t' + char_label + '\n')

    def read_label_text(self, label_filepath):
        res_dict = {}
        for line in open(label_filepath, 'r', encoding='utf-8'):
            # 数据格式：
            # 右髋部	21	23	身体部位
            # 疼痛	27	28	症状和体征
            # 肿胀	29	30	症状和体征

            # 3.1 基于原始序列根据标注信息，得到实体开始索引、结束索引、中文标签类型
            # TODO strip去掉文本中的多余的空格. 做用：去掉文本左右的空格
            res = line.strip().split('\t')
            # res-->['右髋部', '21', '23', '身体部位']
            # TODO 实际工作中，这里需要加一些try catch，保护性的代码。 可以让大模型改写
            start = int(res[1])
            end = int(res[2])
            label = res[3]
            # 3.2.基于深度学习实现NER 基于labels.json得到中文标签类型到tag转换
            label_tag = self.label_dict.get(label)

            # 3.3 基于标注中的位置信息得到BIO标签
            # 右髋部	21	23	身体部位
            # 21是start , 23是end . 这个中间21、22、23一共是3个索引，对应的3个字符。 所以end是闭合的
            # range是左闭右开的， 所以在进行range操作的时候， end要+1
            for i in range(start, end + 1):
                # 因为我们使用的是BIO标注法，所以如果是开始： 加一个B-前缀
                if i == start:
                    tag = "B-" + label_tag
                # 如果是中间或者结尾， 我们加一个 I-前缀
                else:
                    tag = "I-" + label_tag
                res_dict[i] = tag
        # TODO res_dict : {21:'B-BODY'}， 只有B和I，没有O
        return res_dict


if __name__ == '__main__':
    transfer_data = TransferData()
    print(transfer_data.label_dict)
    print(transfer_data.seq_tag_dict)
    transfer_data.transfer()