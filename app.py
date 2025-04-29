from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Load dataset and train the model once at the beginning
@app.before_first_request
def load_model():
    try:
        # Load dataset
        df = pd.read_excel("office_load_dataset_24hr.xlsx")

        # Convert times to minutes since midnight
        df["Office Start Time"] = pd.to_datetime(df["Office Start Time"], format="%H:%M").dt.hour * 60 + pd.to_datetime(df["Office Start Time"], format="%H:%M").dt.minute
        df["Office End Time"] = pd.to_datetime(df["Office End Time"], format="%H:%M").dt.hour * 60 + pd.to_datetime(df["Office End Time"], format="%H:%M").dt.minute

        # Feature matrix and target
        X = df[["Office Start Time", "Office End Time", "Load During Office Time", "Load After Office Time"]]
        y = df["Action"]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, "model.pkl")

        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model Trained Successfully! Test Accuracy: {accuracy:.2f}")

    except Exception as e:
        print(f"Error during model training: {str(e)}")


# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json(force=True)

        # Extract features
        office_start_time = data['office_start_time']
        office_end_time = data['office_end_time']
        load_during_office_time = data['load_during_office_time']
        load_after_office_time = data['load_after_office_time']

        # Load the trained model
        model = joblib.load("model.pkl")

        # Prepare input for prediction
        input_data = [[office_start_time, office_end_time, load_during_office_time, load_after_office_time]]

        # Make prediction
        prediction = model.predict(input_data)

        # Return prediction as response
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
