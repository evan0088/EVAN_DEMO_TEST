# 导入模块
import jieba


# 初始化文本
content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

# todo:精确模式, 更符合人类的阅读习惯, 适合文本分析
# 调用jieba模块的cut方法, 默认精确分词, jieba自动切分成词, 返回迭代对象
# sentence: 待分割的文本
# cut_all: False,默认值,精确分词
obj = jieba.cut(sentence=content, cut_all=False)
print('obj--->', obj)
# 获取分词结果的列表
# l->list
result1 = jieba.lcut(sentence=content, cut_all=False)
print('result1--->', result1)

# todo:全模式, 将文本中所有可以成词的词语都分开, 会有歧义问题
# cut_all: True,全模式
result2 = jieba.lcut(sentence=content, cut_all=True)
print('result2--->', result2)

# todo:搜索引擎模式, 在精确模式的分词结果基础上再次对长词进行分词
result3 = jieba.lcut_for_search(sentence=content)
# jieba.cut_for_search()
print('result3--->', result3)

# todo:中文繁体分词
content2 = "煩惱即是菩提，我暫且不提"
result4 = jieba.lcut(sentence=content2)
print('result4--->', result4)

# todo:自定义词典分词
"""
格式: word(词) freq(词频,可选) word_tpye(词性,可选)
黑马程序员 10 n
传智教育 3 n
...
"""
# 加载自定义字典, 将分词后还有歧义的词进一步划分
jieba.load_userdict(f='data/userdict.txt')
# 将自定义词加入词典, 缓存中
jieba.add_word("我是")
result5 = jieba.lcut(sentence=content)
print('result5--->', result5)