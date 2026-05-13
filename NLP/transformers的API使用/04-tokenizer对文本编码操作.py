from transformers import BertTokenizer
import torch


text1 = "9月3日举行抗日胜利80周年大阅兵"
# text2 = ["9月3日举行抗日胜利80周年大阅兵"]
text2 = ["9月3日举行抗日胜利80周年大阅兵", "小日本投降了!"]
text3 = [["小日本什么时候投降的?", "1945年9月2日"], ["中国什么时候成立?", "1949年10月1日"]]

# 创建分词器
my_tokenizer = BertTokenizer.from_pretrained('model/bert-base-chinese')

############################### encode() ###############################
# 处理1个样本, 返回一维列表或二维张量对象
# # return_tensors='pt': 返回二维张量对象,  不设置返回一维列表
# encode_result1 = my_tokenizer.encode(text1, return_tensors='pt')
# print('encode_result1--->', encode_result1)
# encode_result2 = my_tokenizer.encode(text2, return_tensors='pt')
# # cls UNK UNK SEP
# print('encode_result2--->', encode_result2)

############################### encode_plus() ###############################
# # 处理1个样本, 返回字典对象
# # return_tensors='pt': 返回二维张量对象,  不设置返回一维列表
# encode_result1 = my_tokenizer.encode_plus(text1, return_tensors='pt')
# print('encode_result1--->', encode_result1)
# encode_result2 = my_tokenizer.encode_plus(text2, return_tensors='pt')
# # cls UNK UNK SEP
# print('encode_result2--->', encode_result2)


############################### batch_encode_plus() ###############################
# 处理多个样本, 返回字典对象
# return_tensors='pt': 返回二维张量对象,  不设置返回一维列表
# encode_result1 = my_tokenizer.batch_encode_plus(text1, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
# print('encode_result1--->', encode_result1)
# encode_result2 = my_tokenizer.batch_encode_plus(text2, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
# # cls UNK UNK SEP
# print('encode_result2--->', encode_result2)
# encode_result3 = my_tokenizer.batch_encode_plus(text3, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
# print('encode_result3--->', encode_result3)

############################### tokenizer() ###############################
# encode_plus & batch_encode_plus
# 处理多个样本, 返回字典对象
# return_tensors='pt': 返回二维张量对象,  不设置返回一维列表
encode_result1 = my_tokenizer(text1, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
print('encode_result1--->', encode_result1)
encode_result2 = my_tokenizer(text2, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
# cls UNK UNK SEP
print('encode_result2--->', encode_result2)
encode_result3 = my_tokenizer(text3, return_tensors='pt', truncation=True, padding='max_length', max_length=30)
print('encode_result3--->', encode_result3)