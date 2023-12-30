import sys
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def load_data(filename='data/california_housing_prepared.csv'):
    """Load test data from a CSV file."""
    data = pd.read_csv(filename)
    X_test = data.drop('TARGET', axis=1)
    y_test = data['TARGET']
    return X_test, y_test

def load_model(filename='models/rf_model.joblib'):
    """Load a trained model from a file."""
    return joblib.load(filename)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set and print the Mean Squared Error."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def main(cutoff=0.1):
    X_test, y_test = load_data()
    model = load_model()
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse} (cutoff={cutoff})')
    if mse > cutoff:
        sys.exit(1)


if __name__ == '__main__':
    main()
