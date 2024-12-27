import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os
import pandas as pd
import os

# Define the file path
data_file_path = r"D:\Projects\stock_market_prediction\data\processed\AAPL_processed_data.csv"

# Check if the file exists
if os.path.exists(data_file_path):
    print("File exists.")
else:
    print("File does not exist. Check the file path.")

# Attempt to load the file
try:
    data = pd.read_csv(data_file_path, parse_dates=['Date'])
    print("File loaded successfully.")
    print(data.head())
except Exception as e:
    print(f"Error while loading file: {e}")
data_file_path = r"D:\Projects\stock_market_prediction\data\processed\AAPL_processed_data.csv"

def load_data():
    print(data_file_path)  # Accessible if defined globally



# Load processed data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'])
        data.set_index('Date', inplace=True)  # Set 'Date' as the index
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the file path.")
        raise
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        raise


# Split the data into training and testing sets
def split_data(data):
    try:
        # Feature columns
        X = data[['High', 'Low', 'Open', 'Volume']]
        # Target column
        y = data['Close']
        
        # Split into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test
    except KeyError as e:
        print(f"Missing required columns in data: {e}")
        raise


# Train the model
def train_model(X_train, y_train):
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("Model training completed.")
        return model
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        raise


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print("Model Evaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        return y_pred
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        raise


# Save the model
def save_model(model, output_dir, model_filename="stock_price_model.pkl"):
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        raise


# Save predictions
def save_predictions(y_test, y_pred, output_path):
    try:
        predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}.")
    except Exception as e:
        print(f"An error occurred while saving predictions: {e}")
        raise


# Main function to execute the training process
def main():
    # File paths
    data_file_path = r"D:\Projects\stock_market_prediction\data\processed\AAPL_processed_data.csv"
    model_output_dir = r"D:\Projects\stock_market_prediction\models"
    predictions_output_path = r"D:\Projects\stock_market_prediction\data\predictions.csv"

    # Debugging: Check data file path
    print(f"Looking for data file at: {data_file_path}")
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}. Please check the file path.")
        return

    # Load data
    data = load_data(data_file_path)

    # Splitting data
    X_train, X_test, y_train, y_test = split_data(data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)

    # Save the trained model
    save_model(model, model_output_dir)

    # Save the predictions
    save_predictions(y_test, y_pred, predictions_output_path)

if __name__ == "__main__":
    main()
