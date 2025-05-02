import os
import json
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import numpy as np
import pickle
from flask import Flask, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load Firebase credentials from environment variable
firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

if not firebase_credentials_json:
    raise ValueError("Firebase credentials JSON not found in environment variable")

cred_dict = json.loads(firebase_credentials_json)

# Initialize Firebase
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'
})

# Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running! Use GET /auto-shutoff"})

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
            if key in ['B1', 'B2', 'B3'] and value == "1":
                start_time_str = usage_ref.child(key).get()

                if not start_time_str:
                    usage_ref.child(key).set(now.isoformat())
                    print(f"Initial ON time for {key}: {now.isoformat()}")
                    continue

                start_time = datetime.fromisoformat(start_time_str)
                elapsed = (now - start_time).total_seconds() / 60
                print(f"Elapsed time for {key}: {elapsed:.2f} minutes")

                if elapsed >= 2:
                    duration = 2
                    load_during = 1
                    load_after = 1  # Tum chaaho to yeh logic improve kar sakte ho based on actual sensor reading

                    features = [duration, load_during, load_after]
                    features_np = np.array([features], dtype=np.float64)

                    print(f"Input features for {key}: {features}")
                    prediction = model.predict(features_np)[0]
                    print(f"Prediction for {key}: {prediction}")

                    if prediction == 1:
                        ref.child(key).set("0")
                        usage_ref.child(key).delete()
                        print(f"🔴 {key} turned OFF by ML model.")
                    else:
                        print(f"✅ {key} remains ON.")
                else:
                    print(f"⏳ {key} ON for only {elapsed:.2f} minutes. Waiting...")
            else:
                usage_ref.child(key).delete()

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
