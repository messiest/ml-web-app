import os
from joblib import load

import pandas as pd
from flask import Flask, request, jsonify


app = Flask(__name__) # create a Flask app

model = load(os.path.join('jar', 'iris_classifier.joblib'))

@app.route("/predict", methods=['POST'])
def predict():
    classes = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
    data = request.get_json()  # features in JSON

    print(data, type(data))

    df = pd.DataFrame(data, index=[0])
    print(df.head())

    pred = model.predict(df.values)

    return jsonify(classes[pred[0]])


if __name__ == '__main__':
    app.run(port=3000, debug=True)
