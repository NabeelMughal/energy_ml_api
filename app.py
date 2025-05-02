from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
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

# Load trained ML model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running! Use /auto-shutoff or /predict"})

@app.route("/auto-shutoff", methods=["GET"])
def auto_shutoff():
    try:
        ref = db.reference('/')
        appliances = ref.get()
        if not appliances:
            return jsonify({"error": "No appliance data found"}), 404

        usage_ref = db.reference('Appliance Usage Time')
        now = datetime.utcnow()
        response_data = {}

        for key, value in appliances.items():
            if key in ['B1', 'B2', 'B3'] and value == "1":
                start_time_str = usage_ref.child(key).get()
                if not start_time_str:
                    usage_ref.child(key).set(now.isoformat())
                    print(f"[SET] Start time for {key} = {now.isoformat()}")
                    continue

                start_time = datetime.fromisoformat(start_time_str)
                elapsed = (now - start_time).total_seconds() / 60
                print(f"[DEBUG] {key}: Elapsed time = {elapsed:.2f} minutes")

                if elapsed >= 2:
                    duration = 2
                    load_during = 1
                    load_after = 0
                    time_of_day = 0 if now.hour < 12 else 1
                    week_day = now.weekday()

                    features = {
                        'Office Duration (Minutes)': duration,
                        'Load During Office Time': load_during,
                        'Load After Office Time': load_after,
                        'Time of Day': time_of_day,
                        'Week Day': week_day
                    }

                    prediction = model.predict([list(features.values())])[0]
                    print(f"[ML] {key} - Features: {list(features.values())} => Prediction: {prediction}")

                    if prediction == 1:
                        ref.child(key).set("0")
                        usage_ref.child(key).delete()
                        print(f"[OFF] 🔴 {key} turned OFF by ML.")
                    else:
                        print(f"[INFO] ✅ {key} stays ON based on prediction.")
                else:
                    print(f"[WAIT] {key} is ON for only {elapsed:.2f} min")

            else:
                usage_ref.child(key).delete()

        return jsonify({"message": "Auto shut-off check completed."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        office_start = datetime.strptime(data['office_start'], "%H:%M")
        office_end = datetime.strptime(data['office_end'], "%H:%M")
        duration = int((office_end - office_start).seconds // 60)

        features = {
            'Office Duration (Minutes)': duration,
            'Load During Office Time': int(data['load_during']),
            'Load After Office Time': int(data['load_after']),
            'Time of Day': int(data['time_of_day']),
            'Week Day': int(data['week_day'])
        }

        prediction = model.predict([list(features.values())])[0]
        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
