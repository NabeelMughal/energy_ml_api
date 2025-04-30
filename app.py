from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

model = None

# Function to convert time string to minutes
def time_to_minutes(t_str):
    time_obj = datetime.strptime(t_str, "%H:%M")
    return time_obj.hour * 60 + time_obj.minute

# Load and train the model
def load_model():
    global model
    try:
        df = pd.read_excel("office_load_dataset_24hr.xlsx")

        # Convert times to minutes since midnight
        df["Office Start Time"] = df["Office Start Time"].apply(time_to_minutes)
        df["Office End Time"] = df["Office End Time"].apply(time_to_minutes)

        # Features and label
        X = df[["Office Start Time", "Office End Time", "Load During Office Time", "Load After Office Time"]]
        y = df["Action"]

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Save trained model
        joblib.dump(model, "model.pkl")

        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"✅ Model trained. Test Accuracy: {accuracy:.2f}")

    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")

# Train once at startup
load_model()

@app.route('/', methods=['GET'])
def index():
    return "✅ API is running! Use POST /predict with JSON data."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model
        if model is None:
            model = joblib.load("model.pkl")

        data = request.get_json(force=True)

        # Extract input values
        start_time_str = data['office_start_time']
        end_time_str = data['office_end_time']
        load_during = data['load_during_office_time']
        load_after = data['load_after_office_time']

        # Convert times to minutes
        start_minutes = time_to_minutes(start_time_str)
        end_minutes = time_to_minutes(end_time_str)

        # Prediction
        input_data = [[start_minutes, end_minutes, load_during, load_after]]
        prediction = model.predict(input_data)

        return jsonify({
            'prediction': int(prediction[0]),
            # 'explanation': '1 = turn off appliances, 0 = no action'  # Optional
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
