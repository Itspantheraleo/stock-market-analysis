import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Example DataFrame (use your actual stock data here)
data = pd.DataFrame({
    'Close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
})

def train_predict_model(data):
    """
    Train a simple Linear Regression model to predict future stock prices.
    Args:
        data (DataFrame): Stock data with historical prices.
    Returns:
        model: Trained model
    """
    print("Training model...")  # Debugging print

    # Create 'Prediction' column which shifts 'Close' by one row
    data['Prediction'] = data['Close'].shift(-1)
    
    # Drop the last row as it will have NaN in 'Prediction'
    data = data[:-1]
    
    print("Data after shifting (Prediction column):")
    print(data)  # Debugging print to check the data

    # Features (X) are 'Close' prices, and target (y) is 'Prediction'
    X = np.array(data['Close']).reshape(-1, 1)
    y = np.array(data['Prediction'])
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Model trained. Coefficients: {model.coef_}, Intercept: {model.intercept_}")  # Debugging print
    
    return model

def predict_future_price(model, last_close_price):
    """
    Predict future stock price using the trained model.
    Args:
        model: Trained machine learning model.
        last_close_price (float): Last known close price.
    Returns:
        float: Predicted future price.
    """
    print("Making prediction...")  # Debugging print

    future_price = model.predict(np.array([[last_close_price]]))
    
    print(f"Predicted future price: {future_price[0]}")  # Debugging print
    return future_price[0]

# Example Usage
print("Starting the process...")  # Debugging print

model = train_predict_model(data)  # Train the model using the historical data

# Get the last close price from the data
last_close_price = data['Close'].iloc[-1]

# Predict the future price using the trained model
predicted_price = predict_future_price(model, last_close_price)

# Print the results
print(f"Last Close Price: {last_close_price}")
print(f"Predicted Next Day Price: {predicted_price}")
