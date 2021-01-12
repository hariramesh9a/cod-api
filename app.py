from flask import Flask
from flask import Response, request
from predict import predict, LSTM
import json

app = Flask(__name__)


@app.route('/')
def hello():
    user = request.args.get('user')
    amount = request.args.get('amount')
    print(user)
    print(amount)
    dict1 = predict(amount=amount, user=user)
    return Response(json.dumps(dict1), mimetype='application/json')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
