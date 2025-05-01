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
                office_start = now
                office_end = now + timedelta(minutes=2)
                duration = int((office_end - office_start).seconds // 60)

                # Features for ML model (as per your dataset)
                features = [
                    duration,
                    1,  # load_during (assume 1 when ON)
                    0   # load_after (assume 0 before shut-off)
                ]
                features_np = np.array([features])

                prediction = model.predict(features_np)[0]
                print(f"Prediction for {key} => {prediction}")

                if prediction == 1:
                    ref.child(key).set("0")  # Turn off appliance
                    usage_ref.child(key).delete()
                    print(f"🔴 {key} turned OFF by ML model.")
                else:
                    usage_ref.child(key).set(now.isoformat())
            else:
                usage_ref.child(key).delete()

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
