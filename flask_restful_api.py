import os
from joblib import load

from flask import Flask, request, jsonify


app = Flask(__name__) # create a Flask app

model = load(os.path.join('jar', 'iris_classifier.joblib'))

@app.route("/", methods=['POST'])
def predict():
    data = request.get_json()  # features in JSON


if __name__=='__main__':
    app.run(port=3000, debug=True)
