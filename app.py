from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np  # Import numpy
import os

app = Flask(__name__)

# Load the trained ML model
model = pickle.load(open("admission_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect the input data from the form
    data = {
        "GRE Score": float(request.form["gre"]),
        "TOEFL Score": float(request.form["toefl"]),
        "University Rating": float(request.form["rating"]),
        "SOP": float(request.form["sop"]),
        "LOR": float(request.form["lor"]),
        "CGPA": float(request.form["cgpa"]),
        "Research": int(request.form["research"])
    }

    # Convert input data to DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)

    # Check if the prediction output is a single value
    if isinstance(prediction, (list, np.ndarray)):
        prediction = prediction[0]

    result = f"🎯 Predicted Chance of Admission: {prediction:.2f}"

    return render_template("index.html", result=result)

# Required for Render to detect port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)