from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
import time
import firebase_admin
from firebase_admin import credentials, db
import json

app = Flask(__name__)
CORS(app)

model = None
dataset_file = "office_load_dataset_24hr.xlsx"

# Load Firebase credentials from environment variable
cred_data = json.loads(os.environ['FIREBASE_CREDENTIALS_JSON'])
cred = credentials.Certificate(cred_data)

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'
})

def time_to_minutes(t_str):
    time_obj = datetime.strptime(t_str, "%H:%M")
    return time_obj.hour * 60 + time_obj.minute

def load_model():
    global model
    try:
        df = pd.read_excel(dataset_file)

        df["Office Start Time"] = df["Office Start Time"].apply(time_to_minutes)
        df["Office End Time"] = df["Office End Time"].apply(time_to_minutes)

        X = df[["Office Start Time", "Office End Time", "Load During Office Time", "Load After Office Time"]]
        y = df["Action"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, "model.pkl")

        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"✅ Model trained. Test Accuracy: {accuracy:.2f}")
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")

# Train on startup
load_model()

@app.route('/', methods=['GET'])
def index():
    return "✅ Auto ML API is running!"

@app.route('/auto-predict', methods=['GET'])
def auto_predict():
    try:
        global model

        # Step 1: Get initial load status
        ref = db.reference('/')
        snapshot = ref.get()
        b1 = snapshot.get('B1', '0')
        b2 = snapshot.get('B2', '0')
        b3 = snapshot.get('B3', '0')
        load_during = 1.0 if b1 == '1' or b2 == '1' or b3 == '1' else 0.0

        # Step 2: Wait 2 minutes
        time.sleep(120)

        # Step 3: Get load after 2 minutes
        snapshot = ref.get()
        b1_after = snapshot.get('B1', '0')
        b2_after = snapshot.get('B2', '0')
        b3_after = snapshot.get('B3', '0')
        load_after = 1.0 if b1_after == '1' or b2_after == '1' or b3_after == '1' else 0.0

        # Step 4: Get current time as dummy office start/end
        now = datetime.now()
        office_start = now.strftime("%H:%M")
        office_end = (now + pd.Timedelta(minutes=2)).strftime("%H:%M")

        # Step 5: Predict using model
        start_minutes = time_to_minutes(office_start)
        end_minutes = time_to_minutes(office_end)

        input_data = [[start_minutes, end_minutes, load_during, load_after]]
        prediction = model.predict(input_data)[0]

        # Step 6: Update Firebase if prediction == 1
        if prediction == 1:
            ref.update({
                'B1': '0',
                'B2': '0',
                'B3': '0'
            })
            action = "Appliances turned OFF"
        else:
            action = "No action needed"

        # Step 7: Append to dataset
        new_row = {
            "Office Start Time": office_start,
            "Office End Time": office_end,
            "Load During Office Time": load_during,
            "Load After Office Time": load_after,
            "Action": prediction
        }

        df = pd.DataFrame([new_row])

        if os.path.exists(dataset_file):
            df_existing = pd.read_excel(dataset_file)
            df_combined = pd.concat([df_existing, df], ignore_index=True)
            df_combined.to_excel(dataset_file, index=False)
        else:
            df.to_excel(dataset_file, index=False)

        # Step 8: Retrain model
        load_model()

        return jsonify({
            'prediction': int(prediction),
            'action': action
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
