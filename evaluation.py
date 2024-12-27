import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

if not os.path.exists("data/processed/AAPL_processed_data.csv"):
    raise FileNotFoundError("The file 'data/processed/AAPL_processed_data.csv' does not exist. Please run preprocessing.py.")


def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'])

def evaluate_model(model_path, test_data_path):
    # Load the model
    model = joblib.load(model_path)
    
    # Load the test data
    data = load_data(test_data_path)
    X_test = data[['Open', 'High', 'Low', 'Volume']].values
    y_test = data['Close'].values

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], y_test, label="Actual Prices", color="blue")
    plt.plot(data['Date'], predictions, label="Predicted Prices", color="red")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Model Predictions vs. Actual Prices")
    plt.legend()
    plt.show()

# Run evaluation
if __name__ == "__main__":
    evaluate_model("models/stock_price_model.pkl", "data/processed/AAPL_processed_data.csv")
