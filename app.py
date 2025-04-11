from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load("admission_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "GRE Score": float(request.form["gre"]),
            "TOEFL Score": float(request.form["toefl"]),
            "University Rating": float(request.form["rating"]),
            "SOP": float(request.form["sop"]),
            "LOR": float(request.form["lor"]),
            "CGPA": float(request.form["cgpa"]),
            "Research": int(request.form["research"])
        }

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        prediction = round(prediction * 100, 2)

        result = f"🎯 Predicted Chance of Admission: {prediction}%"
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

# For Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
