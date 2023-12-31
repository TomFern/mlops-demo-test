import pandas as pd
import joblib
from flask import Flask, request, jsonify

def load_model(filename='../models/rf_model.joblib'):
    """Load a trained model from a file."""
    return joblib.load(filename)


def make_prediction(model, input_data):
    """Make a prediction using the model and input data."""
    return model.predict(input_data)


def prepare_input_data(sample_input):
    """Prepare the input data for prediction. This should match the format used for training the model."""
    return pd.DataFrame([sample_input])


app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    sample_input = request.json
    print("Serving prediction request: {sample_input}")
    model = load_model()
    input_data = prepare_input_data(sample_input)
    prediction = make_prediction(model, input_data)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
     print("Listening on port 8080")
     app.run(host="0.0.0.0", port=8080)
