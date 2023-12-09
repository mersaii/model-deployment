from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the saved model
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# create a route that manages user request and does sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return jsonify({'sentiment': str(prediction)})

if __name__ == '__main__':
    app.run()
