import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def load_data(filename='data/california_housing_prepared.csv'):
    data = pd.read_csv(filename)
    X = data.drop('TARGET', axis=1)
    y = data['TARGET']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='models/rf_model.joblib'):
    joblib.dump(model, filename)

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    save_model(model)

if __name__ == '__main__':
    main()
