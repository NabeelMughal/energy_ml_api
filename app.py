from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
import requests
import os
import json
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Firebase initialize
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
creds_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(creds_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'
})

# Load trained ML model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running! Use GET /auto-shutoff or POST /predict"})

@app.route("/auto-shutoff", methods=["GET"])
def auto_shutoff():
    try:
        ref = db.reference('/')
        appliances = ref.get()

        if not appliances:
            return jsonify({"error": "No appliance data found in Firebase."}), 404

        usage_ref = db.reference('Appliance Usage Time')
        now = datetime.utcnow()

        for key, value in appliances.items():
            if key in ['B1', 'B2', 'B3'] and value == "1":
                start_time_str = usage_ref.child(key).get()

                if not start_time_str:
                    usage_ref.child(key).set(now.isoformat())
                    print(f"[INFO] {key} turned ON first time at {now.isoformat()}")
                    continue

                start_time = datetime.fromisoformat(start_time_str)
                elapsed = (now - start_time).total_seconds() / 60

                print(f"[DEBUG] {key}: Elapsed time = {elapsed:.2f} minutes")

                if elapsed >= 2:
                    features_dict = {
                        "Office Duration (Minutes)": 2,
                        "Load During Office Time": 1,
                        "Load After Office Time": 0,
                        "Time of Day": 1 if now.hour >= 12 else 0,
                        "Week Day": now.weekday()
                    }

                    input_df = pd.DataFrame([features_dict])
                    prediction = model.predict(input_df)[0]
                    print(f"[ML] {key} - Features: {list(features_dict.values())} => Prediction: {prediction}")

                    if prediction == 1:
                        ref.child(key).set("0")
                        usage_ref.child(key).delete()
                        print(f"[ACTION] 🔴 {key} turned OFF based on ML model.")
                    else:
                        print(f"[INFO] ✅ {key} stays ON based on prediction.")
                else:
                    print(f"[WAIT] ⏳ {key} ON for only {elapsed:.2f} minutes. Waiting...")

            else:
                usage_ref.child(key).delete()
                print(f"[CLEANUP] {key} is OFF — usage time deleted.")

        return jsonify({"status": "Auto-shutoff check completed."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        office_start = datetime.strptime(data['office_start'], "%H:%M")
        office_end = datetime.strptime(data['office_end'], "%H:%M")
        duration = int((office_end - office_start).seconds // 60)

        features_dict = {
            "Office Duration (Minutes)": duration,
            "Load During Office Time": int(data['load_during']),
            "Load After Office Time": int(data['load_after']),
            "Time of Day": int(data['time_of_day']),
            "Week Day": int(data['week_day'])
        }

        input_df = pd.DataFrame([features_dict])
        prediction = model.predict(input_df)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
