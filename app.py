from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
import requests
import os
import json
from datetime import datetime, timedelta
import joblib
import pandas as pd  # ✅ Added for feature names
import numpy as np

app = Flask(__name__)

# Initialize Firebase
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
creds_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(creds_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'
})

# Load trained ML model once
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
        response_data = {}

        for key, value in appliances.items():
            if key in ['B1', 'B2', 'B3'] and value == "1":  # Appliance is ON
                start_time_str = usage_ref.child(key).get()

                if not start_time_str:
                    usage_ref.child(key).set(now.isoformat())  # First time ON, store current time
                    print(f"[INIT] {key} turned ON at {now.isoformat()}")
                    continue

                start_time = datetime.fromisoformat(start_time_str)
                elapsed = (now - start_time).total_seconds() / 60
                print(f"[DEBUG] {key}: Elapsed time = {elapsed:.2f} minutes")

                if elapsed >= 2:
                    duration = 2
                    load_during = 1
                    load_after = 1 if value == "1" else 0  # ✅ Now based on current Firebase value
                    time_of_day = 1 if now.hour >= 12 else 0
                    week_day = now.weekday()

                    features_df = pd.DataFrame([{
                        "duration": duration,
                        "load_during": load_during,
                        "load_after": load_after,
                        "time_of_day": time_of_day,
                        "week_day": week_day
                    }])

                    prediction = model.predict(features_df)[0]
                    print(f"[ML] {key} - Features: {features_df.values.tolist()[0]} => Prediction: {prediction}")

                    if prediction == 1:
                        ref.child(key).set("0")
                        usage_ref.child(key).delete()
                        print(f"[ACTION] 🔴 {key} turned OFF by ML model.")
                        response_data[key] = "Turned OFF"
                    else:
                        print(f"[INFO] ✅ {key} stays ON based on prediction.")
                        response_data[key] = "Stayed ON"
                else:
                    print(f"[WAIT] ⏳ {key} has been ON for {elapsed:.2f} mins. Waiting 2 mins...")
                    response_data[key] = f"Waiting ({elapsed:.2f} mins)"

            else:
                usage_ref.child(key).delete()
                print(f"[CLEANUP] {key} is OFF — usage time deleted.")
                response_data[key] = "Already OFF"

        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        office_start = datetime.strptime(data['office_start'], "%H:%M")
        office_end = datetime.strptime(data['office_end'], "%H:%M")
        duration = int((office_end - office_start).seconds // 60)

        features_df = pd.DataFrame([{
            "duration": duration,
            "load_during": int(data['load_during']),
            "load_after": int(data['load_after']),
            "time_of_day": int(data['time_of_day']),
            "week_day": int(data['week_day'])
        }])

        prediction = model.predict(features_df)[0]
        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
