from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, db
import requests
import os
import json
from datetime import datetime, timedelta

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

                # --- Predict using ML model ---
                # You can change these inputs according to what your model expects
                payload = {
                    "office_start": now.strftime("%H:%M"),
                    "office_end": (now.replace(minute=now.minute + 2)).strftime("%H:%M"),
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
                    usage_ref.child(key).set(now.isoformat())  # Update usage time

            elif key in ['B1', 'B2', 'B3']:  # If appliance is OFF
                usage_ref.child(key).delete()  # Remove usage time from database

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

        # Dummy model for illustration (you can load your actual model instead)
        X = [[2, 1, 0], [2, 0, 1], [2, 1, 1], [2, 0, 0]]
        y = [0, 1, 1, 0]
        model = LogisticRegression().fit(X, y)
        prediction = model.predict([features])[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Helper function to check if 2 minutes have passed since the appliance was turned on
def check_usage_time_and_turn_off(appliance_id):
    appliance_usage_time = get_appliance_usage_time_from_db(appliance_id)  # Get from DB
    
    if appliance_usage_time:
        appliance_usage_time = datetime.strptime(appliance_usage_time, "%Y-%m-%dT%H:%M:%S.%f")
        current_time = datetime.utcnow()  # Get current time

        # Check if 2 minutes have passed
        if current_time - appliance_usage_time >= timedelta(minutes=2):
            reset_appliance_usage_time(appliance_id)  # Reset appliance usage time to 0
            turn_off_appliance_relay(appliance_id)  # Turn off appliance via relay
            return jsonify({"message": "Appliance turned off due to 2 minutes timeout."})
        else:
            return jsonify({"message": "Appliance still within 2 minutes."})
    else:
        return jsonify({"message": "Appliance not found in database."})

# Helper functions to interact with DB and control relay
def get_appliance_usage_time_from_db(appliance_id):
    # Fetch appliance usage time from DB (for time calculation)
    usage_ref = db.reference('Appliance Usage Time')
    usage_time = usage_ref.child(appliance_id).get()
    return usage_time

def reset_appliance_usage_time(appliance_id):
    # Update DB to set appliance usage time to 0
    usage_ref = db.reference('Appliance Usage Time')
    usage_ref.child(appliance_id).set(None)  # Remove usage time from DB

def turn_off_appliance_relay(appliance_id):
    # Send signal to Arduino to turn off the relay (and appliance)
    pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
