import torch

# 创建一个张量,然后数据类型转换
data = torch.randint(0, 10, [2, 5])
print(data)
print(data.dtype)  # torch.int64
print('===========================================')
# TODO 张量.类型函数()  强烈推荐 使用
# 这里涉及深拷贝,要么创建一个新的张量,要么使用张量.type.dtype()
print(data.short())
print(data.int())
print(data.long(), data.long().dtype)
print(data.half())
print(data.float(), data.float().dtype)
print(data.double())
print('==========================================')
# TODO 张量.type(指定类型)
# 类型转换方式1  torch.小写类型名
print(data.type(torch.short))
print(data.type(torch.int))
print(data.type(torch.long),  data.type(torch.long).dtype)
print(data.type(torch.half))
print(data.type(torch.float), data.type(torch.float).dtype)
print(data.type(torch.double))
print('--------------------------')
# 类型转换方式2 torch.int位数/torch.float位数
print(data.type(torch.int16))
print(data.type(torch.int32))
print(data.type(torch.int64))
print(data.type(torch.float16))
print(data.type(torch.float32))
print(data.type(torch.float64))
print('--------------------------')
# 类型转换方式3 torch.大写类型Tensor  有警告,了解即可
print(data.type(torch.ShortTensor))
print(data.type(torch.IntTensor))
print(data.type(torch.LongTensor))
print(data.type(torch.HalfTensor))
print(data.type(torch.FloatTensor))
print(data.type(torch.DoubleTensor))
