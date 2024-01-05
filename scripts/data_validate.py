import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from os import path

# update this values according at your dataset
data_lines = 20640
data_columns = ['AveBedrms', 'AveOccup', 'AveRooms', 'HouseAge', 
                'Latitude', 'Longitude', 'MedInc', 'Population', 'TARGET']


def load_data(filename=path.join('data', 'california_housing_prepared.csv')):
    data = pd.read_csv(filename)
    return data


def validate(data):
    ok = True
    err = []

    if data.duplicated().sum() > 0:
        ok = False
        err.append('duplicate data points found')

    desc = data.describe()
    print(desc)
    for i in range(0, len(data_columns)):
        col_name = data_columns[i]
        if not col_name in desc:
            ok = False
            err.append(f'expected column "{col_name}" is missing')
        else:
            non_empty_lines = data[col_name].count()
            if  non_empty_lines != data_lines:
                ok = False
                err.append(f'expected column {col_name} to have {data_lines} entries but found {lines} instead')

    return ok, err


def data_reports(data, out_dir='metrics'):

    with open(path.join(out_dir, 'data_report.txt'), 'w') as f:
        sys.stdout = f  # Redirect the print output to file
        print("Data Shape:", data.shape)
        print("\nFirst 5 rows:\n", data.head())
        print("\nData Types and Missing Values:\n", data.info())
        print("\nStatistical Summary:\n", data.describe())
        print("\nNumber of Duplicate Rows:", data.duplicated().sum())
        sys.stdout = sys.__stdout__  # Reset the standard output to original

    # plot data statistics
    data.hist(bins=50, figsize=(20,15))
    plt.savefig(path.join(out_dir,'data_histogram.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.savefig(path.join(out_dir,'data_correlation_matrix.png'))
    plt.close()


def main():
    data = load_data()
    data_reports(data)
    
    ok, err = validate(data)
    if not ok:
        print('ERROR. Data validation failed')
        for i in range(0, len(err)):
            print(f'  - {err[i]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
