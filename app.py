from flask import Flask, render_template, request, send_from_directory
import sqlite3
import pandas as pd
from waitress import serve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import os
import matplotlib

matplotlib.use('Agg')

# Get the current working directory and join it with the template folder path
current_dir = os.getcwd()  # Gets the current working directory
template_folder = os.path.join(current_dir, 'frontend', 'templates')
static_folder = os.path.join(current_dir, 'frontend', 'static')

# Initialize Flask app with the dynamically set template folder
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

# Function to preprocess the data for LSTM
def preprocess_data(data, time_step=60):
    print(f"Preprocessing data for LSTM with time_step={time_step}")
    print("Initial data:")
    print(data.head())

    data = data[['date', 'close']]
    data.set_index('date', inplace=True)

    # Normalize the closing prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']].values)

    # Prepare data for LSTM
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshaping X to be 3D as required by LSTM (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print(f"Data preprocessed. Shapes - X: {X.shape}, y: {y.shape}")
    return X, y, scaler

# Function to build and train the LSTM model
def build_lstm_model(X_train, y_train):
    print(f"Building and training LSTM model with X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Predicting the next closing price
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=5, batch_size=32)
    print("LSTM model trained successfully")
    return model

# Function to make predictions using the trained model
def predict_stock_price(model, X_test, scaler):
    print(f"Predicting stock prices with X_test shape: {X_test.shape}")
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    print("Prediction completed")
    return predicted_prices

# Function to fetch data from SQLite database
def fetch_data_from_db(ticker):
    print(f"Fetching data from database for ticker: {ticker}")
    try:
        conn = sqlite3.connect('stock_data4.db')
        query = f"SELECT * FROM stock_prices WHERE ticker = ircon.NS"
        data = pd.read_sql(query, conn, params=(ticker,), parse_dates=['date'])
        conn.close()

        if data.empty:
            print(f"No data found for ticker: {ticker}")
            return pd.DataFrame()  # Return empty DataFrame

        print(f"Data fetched successfully for {ticker}")
        print(data.head())
        return data
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return pd.DataFrame()

# Function to plot and save stock data
def plot_stock_data(data, ticker):
    print(f"Plotting stock data for ticker: {ticker}")
    if data.empty:
        print("No data to plot")
        return None, None

    # Ensure 'date' is set as the index for plotting
    data.set_index('date', inplace=True)

    # Create output directory for plots if not exists
    output_dir = "./frontend/static/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot Close Price Trend
    close_plot_path = os.path.join(output_dir, f"{ticker}_close_price.png")
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'], label="Close Price", color='blue')
    plt.title(f"{ticker} - Close Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid()
    plt.legend()
    plt.savefig(close_plot_path)
    plt.close()
    print(f"Close price plot saved at: {close_plot_path}")

    # Plot Trading Volume
    volume_plot_path = os.path.join(output_dir, f"{ticker}_volume.png")
    plt.figure(figsize=(12, 6))
    plt.bar(data.index, data['volume'], color='orange')
    plt.title(f"{ticker} - Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.grid()
    plt.savefig(volume_plot_path)
    plt.close()
    print(f"Volume plot saved at: {volume_plot_path}")

    return close_plot_path, volume_plot_path

# Route: Homepage
@app.route('/')
def index():
    print("Rendering homepage")
    return render_template('index.html')

# Route: Get Stock Data
@app.route('/get_data', methods=['POST'])
def get_data():
    tickers = request.form['ticker'].split(',')  # Split the tickers by commas
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    print(f"Received request for tickers: {tickers}, start_date: {start_date}, end_date: {end_date}")

    stock_data = {}
    predictions = {}
    plot_paths = {}  # Store plot paths for each ticker

    for ticker in tickers:
        ticker = ticker.strip()  # Remove any leading/trailing spaces
        data = fetch_data_from_db(ticker)  # Fetch data from the database

        if data.empty:
            print(f"No data available for ticker: {ticker}")
            continue

        # Preprocess data for LSTM
        X, y, scaler = preprocess_data(data)

        # Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train the LSTM model
        model = build_lstm_model(X_train, y_train)

        # Make predictions
        predicted_prices = predict_stock_price(model, X_test, scaler)

        # Get plot paths
        close_plot, volume_plot = plot_stock_data(data, ticker)

        # Store the data and plots
        stock_data[ticker] = data
        predictions[ticker] = predicted_prices
        plot_paths[ticker] = (close_plot, volume_plot)

    print("Rendering results page")
    return render_template('results.html', stock_data=stock_data, predictions=predictions, plot_paths=plot_paths,ticker=ticker,data_empty = stock_data['data'].empty if 'data' in stock_data and isinstance(stock_data['data'], pd.DataFrame) else True)

# Start the Flask app using Waitress
if __name__ == '__main__':
    print("Starting Flask app with Waitress")
    serve(app, host='0.0.0.0', port=8000)
