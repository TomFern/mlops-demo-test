import pandas as pd

def load_data(filename='data/california_housing.csv'):
    return pd.read_csv(filename)

def preprocess_data(data):
    # preprocess/validation steps go here
    return data

def save_data(data, filename='data/california_housing_prepared.csv'):
    data.to_csv(filename, index=False)

def main():
    data = load_data()
    data = preprocess_data(data)
    save_data(data)

if __name__ == '__main__':
    main()
