from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route('/')
def index():
    return "Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [[
        data["office_start_time"],
        data["office_end_time"],
        data["load_during_office"],
        data["load_after_office"]
    ]]
    prediction = model.predict(features)
    return jsonify({"prediction": prediction[0]})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
