from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
import requests
import os
import json
from datetime import datetime
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
        now = datetime.now()  # Local time instead of UTC

        response_data = {}

        for key, value in appliances.items():
            if key in ['B1', 'B2', 'B3'] and value == "1":  # Appliance is ON
                start_time_str = usage_ref.child(key).get()

                if not start_time_str:
                    usage_ref.child(key).set(now.isoformat())
                    print(f"[INIT] {key} turned ON at {now.isoformat()}")
                    continue

                start_time = datetime.fromisoformat(start_time_str)
                elapsed = (now - start_time).total_seconds() / 60
                print(f"[DEBUG] {key}: Elapsed time = {elapsed:.2f} minutes")

                if elapsed >= 2:
                    duration = 2
                    load_during = 1
                    load_after = 0  # Default
                    time_of_day = 1 if now.hour >= 12 else 0
                    week_day = now.weekday()

                    features = [duration, load_during, load_after, time_of_day, week_day]
                    features_np = np.array([features])

                    prediction = model.predict(features_np)[0]
                    print(f"[ML] {key} - Features: {features} => Prediction: {prediction}")

                    if prediction == 1:
                        ref.child(key).set("0")
                        usage_ref.child(key).delete()
                        print(f"[ACTION] 🔴 {key} turned OFF and usage time deleted.")
                    else:
                        print(f"[INFO] ✅ {key} stays ON based on prediction.")
                else:
                    print(f"[WAIT] {key} has only been ON for {elapsed:.2f} mins.")
            else:
                usage_ref.child(key).delete()
                print(f"[CLEANUP] {key} is OFF — usage time deleted.")

        return jsonify({"status": "Auto shut-off check completed."})

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

        features = [
            duration,
            int(data['load_during']),
            int(data['load_after']),
            int(data['time_of_day']),
            int(data['week_day'])
        ]
        features_np = np.array([features])

        prediction = model.predict(features_np)[0]
        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        print(f"[PREDICT ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
