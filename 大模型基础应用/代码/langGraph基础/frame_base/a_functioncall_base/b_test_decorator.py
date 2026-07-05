
def test_decorator(func):
    def wrapper():
        print("在原函数之前执行")
        func()
        print("在原函数之后执行")
    return wrapper

@test_decorator
def say_hello():
    #收到数据 打印
    print("Hello!")
    #得到结果 打印



if __name__ == "__main__":
    say_hello()
