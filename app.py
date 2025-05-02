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

# Convert JSON string to dictionary
cred_dict = json.loads(firebase_credentials_json)

# Initialize Firebase Admin SDK
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'  # Replace with your Firebase Realtime Database URL
})
# Load the trained ML model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running! Use GET /auto-shutoff or POST /predict"})


@app.route("/auto-shutoff", methods=["GET"])
def auto_shutoff():
    try:
        ref = db.reference('/')  # Firebase root reference
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
                    features_np = np.array([features], dtype=np.float64)

                    # Print input features for debugging
                    print(f"Input features for {key}: {features}")
                    
                    prediction = model.predict(features_np)[0]
                    
                    # Print prediction result for debugging
                    print(f"Prediction for {key}: {prediction}")

                    if prediction == 1:
                        print(f"Turning OFF {key} in Firebase")
                        ref.child(key).set("0")
                        usage_ref.child(key).delete()
                        print(f"🔴 {key} turned OFF by ML model.")
                    else:
                        print(f"✅ {key} remains ON.")  # Keep it ON in response
                else:
                    print(f"⏳ {key} ON for only {elapsed:.2f} minutes. Waiting...")

            else:
                usage_ref.child(key).delete()  # If appliance OFF, delete tracking

        return jsonify(response_data)  # Return direct appliance data without predictions

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
