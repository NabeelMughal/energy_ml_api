import firebase_admin
from firebase_admin import credentials, db
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime, timedelta

# Load Firebase credentials
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
creds_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(creds_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'
})

app = Flask(__name__)
CORS(app)
model = None

def time_to_minutes(t_str):
    time_obj = datetime.strptime(t_str, "%H:%M")
    return time_obj.hour * 60 + time_obj.minute

def load_model():
    global model
    try:
        df = pd.read_excel("office_load_dataset_24hr.xlsx")
        df["Office Start Time"] = df["Office Start Time"].apply(time_to_minutes)
        df["Office End Time"] = df["Office End Time"].apply(time_to_minutes)
        X = df[["Office Start Time", "Office End Time", "Load During Office Time", "Load After Office Time"]]
        y = df["Action"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, "model.pkl")
        print(f"✅ Model trained. Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")

load_model()

@app.route('/', methods=['GET'])
def index():
    return "✅ API is running! Use POST /predict or GET /auto-shutoff"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model
        if model is None:
            model = joblib.load("model.pkl")

        data = request.get_json(force=True)
        start_time_str = data['office_start_time']
        end_time_str = data['office_end_time']
        load_during = data['load_during_office_time']
        load_after = data['load_after_office_time']

        start_minutes = time_to_minutes(start_time_str)
        end_minutes = time_to_minutes(end_time_str)

        input_data = [[start_minutes, end_minutes, load_during, load_after]]
        prediction = model.predict(input_data)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/auto-shutoff', methods=['GET'])
def auto_shutoff():
    try:
        ref = db.reference('/')
        appliances = ref.get()

        # Log the fetched data from Firebase
        print(f"Appliances Data Retrieved: {appliances}")

        if not appliances:
            return jsonify({"error": "No appliances data found in Firebase."}), 400

        usage_ref = db.reference('Appliance Usage Time')
        now = datetime.utcnow()

        for key, value in appliances.items():
            # Convert value to integer if it's stored as a string
            value = int(value)  # Convert the value to an integer

            if value == 1:  # Appliance is ON
                on_time_str = usage_ref.child(key).get()
                if on_time_str:
                    on_time = datetime.fromisoformat(on_time_str)
                    if now - on_time >= timedelta(minutes=2):
                        ref.child(key).set(0)
                        usage_ref.child(key).delete()
                        print(f"🔴 {key} turned OFF automatically")
                else:
                    usage_ref.child(key).set(now.isoformat())
            else:
                usage_ref.child(key).delete()

        return jsonify({"message": "Auto shut-off check completed."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
