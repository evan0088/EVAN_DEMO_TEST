import math
# 定义语料库
sentences = [
['I', 'have', 'a', 'pen'],
['He', 'has', 'a', 'book'],
['She', 'has', 'a', 'cat']
]
# 定义语言模型
unigram = {
'I': 1/11,
'have': 1/11,
'a': 3/11,
'pen': 1/11,
'He': 1/11,
'has': 2/11,
'book': 1/11,
'She': 1/11,
'cat': 1/11
}
# 计算困惑度
perplexity = 0
for sentence in sentences:
    sentence_prob = 1
    for word in sentence:
        sentence_prob *= unigram[word]
    temp = -math.log(sentence_prob, 2)/len(sentence)
    perplexity+=2**temp
perplexity = perplexity/len(sentences)
print('困惑度为：', perplexity)