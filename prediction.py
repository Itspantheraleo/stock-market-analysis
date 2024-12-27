import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def predict_prices(model, input_data):
    # Standardize input data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_data)
    predictions = model.predict(scaled_data)
    return predictions

def main():
    # Load the trained model
    model = load_model("models/stock_price_model.pkl")

    # Example input for prediction
    new_data = pd.DataFrame({
        "Open": [150.0, 152.0],
        "High": [155.0, 156.0],
        "Low": [148.0, 149.0],
        "Volume": [1_000_000, 900_000]
    })

    # Predict stock prices
    predictions = predict_prices(model, new_data.values)
    print("Predicted Prices:", predictions)

if __name__ == "__main__":
    main()
