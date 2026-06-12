import os

if __name__ == '__main__':
    path = '/LSTM_CRF/data_origin'

    for root, dirs, files in os.walk(path):
        print("*" * 100)
        print(f'root --> {root}')
        print(f'dirs --> {dirs}')
        print(f'files --> {files}')
