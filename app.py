import firebase_admin
from firebase_admin import credentials, db
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime, timedelta

# Load Firebase credentials
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
creds_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(creds_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'
})

app = Flask(__name__)
CORS(app)
model = None

def time_to_minutes(t_str):
    time_obj = datetime.strptime(t_str, "%H:%M")
    return time_obj.hour * 60 + time_obj.minute

def load_model():
    global model
    try:
        df = pd.read_excel("office_load_dataset_24hr.xlsx")
        df["Office Start Time"] = df["Office Start Time"].apply(time_to_minutes)
        df["Office End Time"] = df["Office End Time"].apply(time_to_minutes)
        X = df[["Office Start Time", "Office End Time", "Load During Office Time", "Load After Office Time"]]
        y = df["Action"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, "model.pkl")
        print(f"✅ Model trained. Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")

load_model()

@app.route('/', methods=['GET'])
def index():
    return "✅ API is running! Use POST /predict or GET /auto-shutoff"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model
        if model is None:
            model = joblib.load("model.pkl")

        data = request.get_json(force=True)
        start_time_str = data['office_start_time']
        end_time_str = data['office_end_time']
        load_during = data['load_during_office_time']
        load_after = data['load_after_office_time']

        start_minutes = time_to_minutes(start_time_str)
        end_minutes = time_to_minutes(end_time_str)

        input_data = [[start_minutes, end_minutes, load_during, load_after]]
        prediction = model.predict(input_data)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/auto-shutoff', methods=['GET'])
def auto_shutoff():
    try:
        # Reference to the root of Firebase DB
        ref = db.reference('/')  
        
        # Get the appliance states (B1, B2, B3)
        appliances = ref.get()  # This will fetch everything from the root, including Buttons and Appliance Usage Time

        # Debug log to check data
        print(f"Appliances Data Retrieved: {appliances}")

        # Get the 'Appliance Usage Time' data
        usage_ref = ref.child('Appliance Usage Time')
        now = datetime.utcnow()

        # Check if the appliance states exist in Firebase
        if 'B1' not in appliances or 'B2' not in appliances or 'B3' not in appliances:
            return jsonify({"error": "No appliances data found in Firebase."})

        # Iterate over the appliances to check if they need to be turned off
        for key in ['B1', 'B2', 'B3']:
            value = appliances.get(key)
            if value == "1":  # Appliance is ON
                # Get the appliance's on-time from 'Appliance Usage Time'
                on_time_str = usage_ref.child(key).get()
                if on_time_str:
                    on_time = datetime.fromisoformat(on_time_str)  # Convert to datetime object
                    if now - on_time >= timedelta(minutes=2):  # If 2 minutes have passed
                        ref.child(key).set("0")  # Turn off the appliance (set it to 0)
                        usage_ref.child(key).delete()  # Remove the usage time entry from Firebase
                        print(f"🔴 {key} turned OFF automatically")
                else:
                    # If no on-time entry exists, set the current time as the appliance on-time
                    usage_ref.child(key).set(now.isoformat())

            else:
                # If the appliance is OFF, delete its usage time entry
                usage_ref.child(key).delete()

        return jsonify({"message": "Auto shut-off check completed."})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
