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

        for key, value in appliances.items():
            if key in ['B1', 'B2', 'B3'] and value == "1":  # Appliance is ON
                # Time after 2 minutes
                office_start = now
                office_end = now + timedelta(minutes=2)
                duration = int((office_end - office_start).seconds // 60)  # 2 minutes

                # Assuming that 'Load During' and 'Load After' are set in Firebase (change based on actual data)
                load_during = 1  # Appliance is ON, so Load During = 1
                load_after = 0   # Load After = 0 because we're checking shut-off immediately after 2 minutes

                # Additional features, assume time_of_day and week_day as an example
                time_of_day = 1 if now.hour >= 12 else 0  # Example: 0 for morning, 1 for evening
                week_day = now.weekday()  # Weekday as an integer (0 = Monday, 6 = Sunday)

                # Create features for the ML model (5 features)
                features = [
                    duration,     # Office Duration (Minutes)
                    load_during,  # Load During Office Time
                    load_after,   # Load After Office Time
                    time_of_day,  # Time of Day (e.g., morning/evening)
                    week_day      # Weekday (0 = Monday, 6 = Sunday)
                ]
                features_np = np.array([features])

                # Make prediction using the model
                prediction = model.predict(features_np)[0]
                print(f"Prediction for {key} => {prediction}")

                # If model predicts that appliance should be turned off
                if prediction == 1:
                    ref.child(key).set("0")  # Turn off appliance (set value to 0)
                    usage_ref.child(key).delete()  # Remove usage time data
                    print(f"🔴 {key} turned OFF by ML model.")
                else:
                    usage_ref.child(key).set(now.isoformat())  # Keep usage time if still ON
            else:
                usage_ref.child(key).delete()  # If appliance is OFF, delete usage time

        return jsonify({"message": "Auto shut-off ML prediction check completed."})

    except Exception as e:
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
            int(data['load_after'])
        ]
        features_np = np.array([features])

        prediction = model.predict(features_np)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
