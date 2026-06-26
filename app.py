from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("heart risk data.csv")

# Train Model
X = df[["Age", "Cholesterol", "BP", "HeartRate"]]
Y = df["Disease"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, Y_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    chol = int(request.form["chol"])
    bp = int(request.form["bp"])
    hr = int(request.form["hr"])

    new_data = pd.DataFrame(
        [[age, chol, bp, hr]],
        columns=["Age", "Cholesterol", "BP", "HeartRate"]
    )

    probability = model.predict_proba(new_data)[0][1]
    risk_percent = round(probability * 100, 2)

    if risk_percent >= 50:
        prediction = "⚠ High Risk of Heart Disease"
    else:
        prediction = "✅ No Significant Risk"

    return render_template(
        "index.html",
        prediction=prediction,
        risk=risk_percent
    )

if __name__ == "__main__":
    app.run(debug=True)