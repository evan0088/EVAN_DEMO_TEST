"""
需求：基于规则实现ner任务：从目标文本中提取机构名
思路步骤：
1. 准备: 关键词词表,语料
2.基于深度学习实现NER. 使用带词性标注的分词工具(jieba.posseg)对语料进行分词
3. 筛选分词中分词词性是ns(地名)的分词，并基于关键词进行B+OE序列标注。
其中，E代表关键词, S代表关键词地名，O表示其他词
    3.1 初始化单词列表words和特征列表features
    3.2.基于深度学习实现NER 遍历每个单词，对单词进行判断，并记录到特征列表features中:
        ① 如果在关键词词表中，标记为E
        ② 如果是地名关键词，标记为S
        ③ 如果非以上两种情况，标记为O
4. 对标注好的序列结果(特征列表)中的序列进行特征数据进行正则匹配，匹配满足：'S+O*E+'的序列：
       ① 至少一个地名（S） ②零个或多个普通词（O）③ 至少一个机构结尾词（E）
5. 基于匹配到的内容(起始索引)，将分词进行拼接，得到机构名
"""
import re
import jieba.posseg as pseg

# 1. 准备: 关键词词表,语料
org_tag = ['公司', '有限公司', '大学', '政府', '人民政府', '总局']


def extract_org(text):
    # 2.基于深度学习实现NER. 使用带词性标注的分词工具(jieba.posseg)对语料进行分词
    word_flags = pseg.lcut(text)
    # 3. 筛选分词中分词词性是ns(地名)的分词，并基于关键词进行B+OE序列标注。
    words = []
    # 3.1 初始化单词列表words和特征列表features
    # 3.2.基于深度学习实现NER 遍历每个单词，对单词进行判断，并记录到特征列表features中:
    features = []
    for word, flag in word_flags:
        words.append(word)
        # 其中，E代表关键词, S代表关键词地名，O表示其他词
        #  ① 如果在关键词词表中，标记为E
        if word in org_tag:
            features.append('E')
        #  ② 如果是地名关键词，标记为S
        else:
            if flag == 'ns':
                features.append('S')
            #  ③ 如果非以上两种情况，标记为O
            else:
                features.append('O')

    # 4. 对标注好的序列结果(特征列表)中的序列进行特征数据进行正则匹配，匹配满足：'S+O*E+'的序列：
    print(words)
    print(features)

    labels = ''.join(features)
    #  ① 至少一个地名（S） ②零个或多个普通词（O）③ 至少一个机构结尾词（E）
    pattern = re.compile('S+O*E+')


    # TODO 获取所有匹配成功的序列： SOOOOOE
    match_label = re.finditer(pattern, labels)
    match_list = []

    # for index, (word, feature) in enumerate(zip(words, features)):
    #     print(f'{index}\t{word}\t{feature}')

    # 5. 基于匹配到的内容(起始索引)，将分词进行拼接，得到机构名
    for match in match_label:
        print(match)
        start = match.start()
        end = match.end()
        match_words = words[start:end]
        print(match_words)
        org_word = ''.join(match_words)
        match_list.append(org_word)

    return match_list

if __name__ == '__main__':
    text = "可在接到本决定书之日起六十日内向中国国家市场监督管理总局申请行政复议,杭州海康威视数字技术股份有限公司."
    match_list = extract_org(text)
    print(match_list)