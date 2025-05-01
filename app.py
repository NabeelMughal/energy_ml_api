from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
import requests
import os
import json
from datetime import datetime, timedelta
import joblib
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
        predictions = {}

        for key, value in appliances.items():
            if key in ['B1', 'B2', 'B3'] and value == "1":  # Appliance is ON
                start_time_str = usage_ref.child(key).get()

                if not start_time_str:
                    usage_ref.child(key).set(now.isoformat())  # First time ON, store current time
                    print(f"Initial ON time for {key}: {now.isoformat()}")
                    continue

                # Parse stored time
                start_time = datetime.fromisoformat(start_time_str)
                elapsed = (now - start_time).total_seconds() / 60
                print(f"Elapsed time for {key}: {elapsed} minutes")

                if elapsed >= 2:  # Only if 2 minutes passed
                    duration = 2
                    load_during = 1
                    load_after = 0  # Default, or update later
                    time_of_day = 1 if now.hour >= 12 else 0
                    week_day = now.weekday()

                    features = [duration, load_during, load_after, time_of_day, week_day]
                    features_np = np.array([features])

                    prediction = model.predict(features_np)[0]
                    predictions[key] = prediction  # Store prediction for each appliance
                    print(f"Prediction for {key} => {prediction}")

                    if prediction == 1:
                        ref.child(key).set("0")
                        usage_ref.child(key).delete()
                        print(f"🔴 {key} turned OFF by ML model.")
                else:
                    print(f"⏳ {key} ON for only {elapsed:.2f} minutes. Waiting...")
            else:
                usage_ref.child(key).delete()  # If appliance OFF, delete tracking

        return jsonify({
            "message": "Auto shut-off ML prediction check completed.",
            "predictions": predictions  # Include predictions in response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        office_start = datetime.strptime(data['office_start'], "%H:%M")
        office_end = datetime.strptime(data['office_end'], "%H:%M")
        duration = int((office_end - office_start).seconds // 60)

        # Ensure all features are included
        features = [
            duration,
            int(data['load_during']),
            int(data['load_after']),
            int(data['time_of_day']),  # time_of_day (morning/evening)
            int(data['week_day'])  # week_day (Monday to Sunday)
        ]
        features_np = np.array([features])

        prediction = model.predict(features_np)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
