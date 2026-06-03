import os

if __name__ == '__main__':
    path = '/Users/itheima/Documents/黑马/讲课/线上直播/狂野1-知识图谱/02-代码/3.4基于深度学习实现NER/LSTM_CRF/data_origin'

    for root, dirs, files in os.walk(path):
        print("*" * 100)
        print(f'root --> {root}')
        print(f'dirs --> {dirs}')
        print(f'files --> {files}')
