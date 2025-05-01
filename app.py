from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, db
import requests
import os
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

# Initialize Firebase
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
creds_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(creds_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'
})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running! Use POST /predict or GET /auto-shutoff"})

@app.route("/auto-shutoff", methods=["GET"])
def auto_shutoff():
    try:
        ref = db.reference('/')
        appliances = ref.get()

        if not appliances:
            return jsonify({"error": "No appliances data found in Firebase."}), 404

        usage_ref = db.reference('Appliance Usage Time')
        now = datetime.utcnow()

        for key, value in appliances.items():
            if key in ['B1', 'B2', 'B3'] and value == "1":  # Appliance is ON
                # Create payload based on current time and +2 minutes
                payload = {
                    "office_start": now.strftime("%H:%M"),
                    "office_end": (now + timedelta(minutes=2)).strftime("%H:%M"),
                    "load_during": 1,
                    "load_after": 0
                }

                prediction_res = requests.post("https://web-production-0ef71.up.railway.app/predict", json=payload)
                prediction = prediction_res.json().get("prediction")

                if prediction == 1:
                    ref.child(key).set("0")  # Turn off the appliance
                    usage_ref.child(key).delete()  # Remove usage time
                    print(f"🔴 {key} turned OFF based on ML prediction")
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
        duration = (office_end - office_start).seconds // 60

        features = [
            duration,
            int(data['load_during']),
            int(data['load_after'])
        ]

        # Dummy ML model (replace with your trained model)
        X = [[2, 1, 0], [2, 0, 1], [2, 1, 1], [2, 0, 0]]
        y = [0, 1, 1, 0]
        model = LogisticRegression().fit(X, y)
        prediction = model.predict([features])[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
