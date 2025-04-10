from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained ML model
model = pickle.load(open("admission_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form data
    data = {
        "GRE Score": float(request.form["gre"]),
        "TOEFL Score": float(request.form["toefl"]),
        "University Rating": float(request.form["rating"]),
        "SOP": float(request.form["sop"]),
        "LOR": float(request.form["lor"]),
        "CGPA": float(request.form["cgpa"]),
        "Research": int(request.form["research"])
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)[0]
    result = f"🎯 Predicted Chance of Admission: {prediction:.2f}"

    return render_template("index.html", result=result)

# Use 0.0.0.0 and a dynamic port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)