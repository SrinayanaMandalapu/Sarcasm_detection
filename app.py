from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)


model = joblib.load('sarcasm_detection_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
# Define a route to serve the HTML page
@app.route("/")
def index():
    return render_template("index.html")

# Define a route to handle the AJAX request for sarcasm detection
@app.route("/predict", methods=["POST"])
def predict():
    sentence = request.form["sentence"]
    output = predict_sarcasm(sentence)
    return render_template('predict.html', output=output)

def predict_sarcasm(sentence):
        sentence_vect = vectorizer.transform([sentence])
        prediction = model.predict(sentence_vect)
        if prediction[0] == 1:
            return "Sarcastic"
        else:
            return "Not Sarcastic"

if __name__ == "__main__":
    app.run(debug=True, port = 5001)
