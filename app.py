from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Train model at startup
def load_model():
    global model
    try:
        # Load dataset
        df = pd.read_excel("office_load_dataset_24hr.xlsx")

        # Convert times to minutes since midnight
        df["Office Start Time"] = pd.to_datetime(df["Office Start Time"], format="%H:%M").dt.hour * 60 + pd.to_datetime(df["Office Start Time"], format="%H:%M").dt.minute
        df["Office End Time"] = pd.to_datetime(df["Office End Time"], format="%H:%M").dt.hour * 60 + pd.to_datetime(df["Office End Time"], format="%H:%M").dt.minute

        # Features and target
        X = df[["Office Start Time", "Office End Time", "Load During Office Time", "Load After Office Time"]]
        y = df["Action"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Save model to file
        joblib.dump(model, "model.pkl")

        # Print accuracy for debugging
        y_pred = model.predict(X_test)
        print(f"Model trained successfully. Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    except Exception as e:
        print(f"Model loading/training failed: {str(e)}")

# Load model on startup
model = None
load_model()

# Root route
@app.route('/')
def home():
    return "API is running! Use POST /predict with JSON data."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # DEBUG PRINTS: Comment out in production
        print("Headers:", request.headers)
        print("Raw Body:", request.data)

        # Read input JSON
        data = request.get_json(force=True)
        print("Parsed JSON:", data)

        # Extract features
        office_start_time = data['office_start_time']
        office_end_time = data['office_end_time']
        load_during_office_time = data['load_during_office_time']
        load_after_office_time = data['load_after_office_time']

        # Load model (optional since already loaded)
        model = joblib.load("model.pkl")

        # Prepare input and predict
        input_data = [[office_start_time, office_end_time, load_during_office_time, load_after_office_time]]
        prediction = model.predict(input_data)

        # Return prediction
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
