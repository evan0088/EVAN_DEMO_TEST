#./redis-server
#./redis-cli

# cache/redis_client.py
# 导入 Redis 客户端
import redis
# 导入 JSON 处理
import json
import time

def getConnection():
    client = redis.Redis(
        host="127.0.0.1",
        port="6379",
        password="1234"
    )
    return client

def insertData():
    client = getConnection()
    client.set("123","张程")
    client.delete("123")
    client.set("123",json.dumps({"姓名":"张程","年龄":27}, ensure_ascii=False))

def getData():
     client = getConnection()
     data = client.get("123")
     print(json.loads(data))

if __name__ == '__main__':
    #insertData()
    getData()