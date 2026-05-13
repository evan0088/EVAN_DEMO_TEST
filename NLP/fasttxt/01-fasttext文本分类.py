# 导入fasttext
# 优点: 文本分类时, 模型训练过程做了高度封装, 我们只需要调整参数即可
import fasttext

def dm01():
    # 使用fasttext的train_supervised方法进行文本分类模型的训练
    model = fasttext.train_supervised(input="data/cooking.train")
    # 模型推理
    print(model.predict("Which baking dish is best to bake a banana bread ?"))
    print(model.predict("Why not put knives in the dishwasher?"))
    # 模型评估
    print(model.test("data/cooking.valid"))

# 数据集清洗优化
def dm02():
    model = fasttext.train_supervised(input="data/cooking.pre.train")
    print(model.test("data/cooking.pre.valid"))

# ...

if __name__ == '__main__':
    # dm01()
    dm02()