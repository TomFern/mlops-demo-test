from sklearn.datasets import fetch_california_housing
import pandas as pd

def download_data():
    california = fetch_california_housing()
    data = pd.DataFrame(california.data, columns=california.feature_names)
    data['TARGET'] = california.target
    return data

def save_data(data, filename='data/california_housing.csv'):
    data.to_csv(filename, index=False)

def main():
    data = download_data()
    save_data(data)

if __name__ == '__main__':
    main()
