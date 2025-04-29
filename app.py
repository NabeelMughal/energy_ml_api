from flask import Flask
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

@app.route('/')
def train_model():
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
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save model
        joblib.dump(model, "model.pkl")

        return f"""
        <h2>Model Trained Successfully!</h2>
        <p><strong>Cross-Validation Accuracy:</strong> {cv_scores.mean():.2f}</p>
        <p><strong>Test Accuracy:</strong> {accuracy_score(y_test, y_pred):.2f}</p>
        <p><strong>Model File Saved:</strong> model.pkl</p>
        """

    except Exception as e:
        return f"<h3>Error occurred:</h3><pre>{str(e)}</pre>"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
