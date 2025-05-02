from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_admin import credentials, db, initialize_app
import pandas as pd
import joblib
from datetime import datetime, timedelta
import pytz
import os

# Flask setup
app = Flask(__name__)
CORS(app)

# Firebase setup
cred = credentials.Certificate({
    "type": os.getenv("FB_TYPE"),
    "project_id": os.getenv("FB_PROJECT_ID"),
    "private_key_id": os.getenv("FB_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FB_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FB_CLIENT_EMAIL"),
    "client_id": os.getenv("FB_CLIENT_ID"),
    "auth_uri": os.getenv("FB_AUTH_URI"),
    "token_uri": os.getenv("FB_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FB_AUTH_PROVIDER"),
    "client_x509_cert_url": os.getenv("FB_CLIENT_CERT")
})
initialize_app(cred, {
    'databaseURL': os.getenv("FB_DB_URL")
})

# Load model
model = joblib.load("model.pkl")

@app.route('/auto-shutoff', methods=['POST'])
def auto_shutoff():
    try:
        ref = db.reference("appliances")
        data = ref.get()

        if not data:
            return jsonify({"error": "No data found in Firebase."}), 404

        now = datetime.now(pytz.timezone("Asia/Karachi"))
        current_hour = now.hour
        current_weekday = now.weekday()

        updates = {}
        for bulb in ["B1", "B2", "B3"]:
            bulb_data = data.get(bulb, {})
            start_time_str = bulb_data.get("time", "")
            status = bulb_data.get("status", 0)

            if not start_time_str or status == 0:
                continue

            # Parse time and calculate duration
            start_time = datetime.strptime(start_time_str, "%H:%M")
            start_time = now.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0)
            duration = (now - start_time).total_seconds() / 60

            # Check if 2 minutes have passed
            if duration >= 2:
                features = pd.DataFrame([{
                    "Load During Office Time": 1,
                    "Load After Office Time": 1,
                    "Office Duration (Minutes)": int(duration),
                    "Time of Day": current_hour,
                    "Week Day": current_weekday
                }])

                action = model.predict(features)[0]

                if action == 1:
                    updates[f"{bulb}/status"] = 0
                    updates[f"{bulb}/time"] = ""

        if updates:
            ref.update(updates)
            return jsonify({"message": "Appliances updated", "updates": updates}), 200
        else:
            return jsonify({"message": "No updates necessary"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
