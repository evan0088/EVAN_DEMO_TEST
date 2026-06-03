

if __name__ == '__main__':
    ids = [1,2,3,4,5,6]
    tokens = ['a','b','c','d','e','f']

    for id, token in zip(ids, tokens):
        print(f'{id}\t{token}')