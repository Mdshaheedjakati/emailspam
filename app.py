from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("models/cv.pkl", "rb") as file:
    cv = pickle.load(file)
clf = pickle.load(open("models/clf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('email-content')  # Corrected field name
    tokenized_email = cv.transform([email])  # Corrected variable name
    prediction = clf.predict(tokenized_email)[0]  # Ensure prediction is extracted properly
    prediction = "🚨 Warning: Potential Spam Detected! 🚨" if prediction == 1 else "Not Spam"
    
    return render_template("index.html", predictions=prediction, text=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # Changed port to avoid conflict
