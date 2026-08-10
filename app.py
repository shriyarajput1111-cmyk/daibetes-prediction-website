from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    data = np.array(values).reshape(1, -1)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Person is Diabetic"
    else:
        result = "Person is Not Diabetic"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)