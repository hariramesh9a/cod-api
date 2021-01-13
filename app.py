from flask import Flask
from flask import Response, request
from predict import predict, LSTM
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello():
    user = request.args.get('user')
    amount = request.args.get('amount')
    print(user)
    print(amount)
    dict1 = predict(amount=amount, user=user)
    return Response(json.dumps(dict1), mimetype='application/json')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
