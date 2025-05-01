import firebase_admin
from firebase_admin import credentials, db
import json
import os
from flask import Flask, jsonify

# Initialize Firebase Admin
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
creds_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(creds_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://energy-monitoring-and-tarif-default-rtdb.firebaseio.com/'
})

app = Flask(__name__)

@app.route('/auto-shutoff', methods=['GET'])
def auto_shutoff():
    try:
        # Fetching data from the root of the database
        ref = db.reference('/')
        appliances = ref.get()

        # Check if appliances data exists
        if not appliances or 'B1' not in appliances or 'B2' not in appliances or 'B3' not in appliances:
            return jsonify({"error": "No appliances data found in Firebase."})
        
        # Print fetched data for debugging
        print(f"Appliances Data Retrieved: {appliances}")

        # Check if any appliance is ON and should be turned off after 2 minutes
        for key, value in appliances.items():
            if key in ['B1', 'B2', 'B3'] and value == "1":  # Appliance is ON
                # Perform the auto-shutoff logic here
                print(f"🔴 {key} will be turned OFF automatically.")
        
        return jsonify({"message": "Auto shut-off check completed."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
