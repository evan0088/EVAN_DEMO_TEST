import jieba.posseg as pseg

# 初始化文本
content = "我爱北京天安门, 天安门上太阳升。"

# 使用jieba分词并标注词性
words_list = pseg.lcut(sentence=content)
print('words--->', words_list)

# 根据实体的词性获取对应的实体信息
named_entities = []
for word, flag in words_list:
    if flag in ['ns', 'nt']:
        named_entities.append((word, flag))

print('named_entities--->', set(named_entities))
