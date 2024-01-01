import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error
from alive_progress import alive_bar


def load_model(filename='models/rf_model.joblib'):
    """Load the trained model from a file."""
    return joblib.load(filename)


def generate_grid(data, feature_x='Longitude', feature_y='Latitude', steps=50):
    """Generate a grid for two features."""
    x_min, x_max = data[feature_x].min(), data[feature_x].max()
    y_min, y_max = data[feature_y].min(), data[feature_y].max()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, steps), 
                                 np.linspace(y_min, y_max, steps))
    return x_grid, y_grid


def predict_on_grid(model, data, x_grid, y_grid, feature_x='Longitude', feature_y='Latitude'):
    """Predict on each point of the grid and compute MSE."""
    mse_values = np.zeros(x_grid.shape)
    total_iterations = len(range(x_grid.shape[0])) * len(range(x_grid.shape[0]))
    with alive_bar(total_iterations) as bar:
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                point_data = data.copy()
                point_data[feature_x] = x_grid[i, j]
                point_data[feature_y] = y_grid[i, j]
                preds = model.predict(point_data.drop('TARGET', axis=1))
                mse_values[i, j] = mean_squared_error(point_data['TARGET'], preds)
                bar()
    return mse_values


def plot_mse(x_grid, y_grid, mse_values, output_filename='metrics/mse_plot.png'):
    """Plot the MSE values over the grid."""
    plt.figure(figsize=(10, 6))
    plt.contourf(x_grid, y_grid, mse_values, cmap='RdBu', levels=100)
    plt.colorbar(label='MSE')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('MSE based on Latitude and Longitude')
    plt.savefig(output_filename)
    plt.close()


def save_mse_values(x_grid, y_grid, mse_values, output_filename='metrics/mse_values.csv'):
    """Save the MSE values in a CSV file."""
    flattened_mse = mse_values.flatten()
    flattened_lat = y_grid.flatten()
    flattened_long = x_grid.flatten()
    mse_df = pd.DataFrame({'Latitude': flattened_lat, 'Longitude': flattened_long, 'MSE': flattened_mse})
    mse_df.to_csv(output_filename, index=False)


def main():
    # ideally, we would use a different dataset to test (not the same one the model was trained on)
    data = pd.read_csv('data/california_housing_prepared.csv')
    model = load_model()
    x_grid, y_grid = generate_grid(data)
    mse_values = predict_on_grid(model, data, x_grid, y_grid)
    plot_mse(x_grid, y_grid, mse_values)
    save_mse_values(x_grid, y_grid, mse_values)


if __name__ == '__main__':
    main()
