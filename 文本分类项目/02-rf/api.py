from flask import Flask,request,Response,json,jsonify
import jieba
import pandas as pd
import pickle
from config import Config
import warnings
from rf_predict_fun import predict

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# 创建一个服务
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# http://127.0.0.1:8003/predict
@app.route('/predict', methods=['POST'])  # 路由 =》类似门牌号
def predict_api():  # 视图函数 =》专门做业务逻辑、提供服务
    # 获取 JSON 输入
    data = request.get_json()   # {text:"......."}
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field in JSON'}), 400

    # 调用预测函数
    result = predict(data)

    # 返回 JSON 结果
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)  # 启动服务
