import sys
import pandas as pd
import joblib

def load_model(filename='models/rf_model.joblib'):
    """Load a trained model from a file."""
    return joblib.load(filename)

def make_prediction(model, input_data):
    """Make a prediction using the model and input data."""
    return model.predict(input_data)

def prepare_input_data(sample_input):
    """Prepare the input data for prediction. This should match the format used for training the model."""
    # Example: convert a dictionary to a pandas DataFrame. Modify this as per your feature set.
    return pd.DataFrame([sample_input])

def main():
    sample_input = {
        'MedInc': 3.5,        # Median income in tens of thousands
        'HouseAge': 35,       # House age in years
        'AveRooms': 6,        # Average number of rooms
        'AveBedrms': 2,       # Average number of bedrooms
        'Population': 800,    # Population in the block
        'AveOccup': 3,        # Average occupancy per household
        'Latitude': 34.2,     # Latitude of the block
        'Longitude': -118.4   # Longitude of the block
    }
    expected_output = 2.0693601

    model = load_model()
    input_data = prepare_input_data(sample_input)
    prediction = make_prediction(model, input_data)
    print(f"Predicted Value: {prediction[0]} (expected {expected_output})")
    if prediction[0] != expected_output:
        sys.exit(1)


if __name__ == '__main__':
    main()
