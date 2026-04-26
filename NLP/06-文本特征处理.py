# todo: n-gram
# n一般1,2,3, 1就是单词本身(独立) 2就是连续两个单词组成一个新词  以此类推

# 2-gram
ngram_range = 2
input_list = [1, 3, 2, 1, 5, 3]  # 假设文本


def demo01():
    # 初始化列表, 存储切片结果
    temp_list = []
    for i in range(ngram_range):
        temp_list.append(input_list[i:])
    print('temp_list--->', temp_list)
    # 使用zip函数将temp_list中的两个列表进行组合, 对应位置的元素进行合并
    result = set(zip(temp_list[0], temp_list[1]))
    print('result--->', result)


if __name__ == '__main__':
    demo01()
