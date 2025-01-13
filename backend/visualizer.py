import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to create the database and table (No changes needed)
def create_database():
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()  

    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Call the create_database function to ensure the table exists
create_database()


# Function to fetch data from the database
def fetch_data_from_db(ticker):
    try:
        conn = sqlite3.connect('stock_data.db')
        query = f"SELECT * FROM stock_prices WHERE ticker = '{ticker}'"
        data = pd.read_sql(query, conn, parse_dates=['date'])
        conn.close()
        
        # Check if the fetched data is empty
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        return data

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return None
    except ValueError as e:
        print(f"Value error: {str(e)}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Function to plot and save stock data
def plot_stock_data(data, ticker):
    if data.empty:
        print(f"No data to plot for {ticker}.")
        return

    # Ensure 'date' is set as the index for plotting
    data.set_index('date', inplace=True)

    # Create output directory for plots if not exists
    output_dir = "./frontend/static/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot Close Price Trend
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'], label="Close Price", color='blue')
    plt.title(f"{ticker} - Close Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid()
    plt.legend()
    close_price_plot_path = os.path.join(output_dir, f"{ticker}_close_price.png")
    plt.savefig(close_price_plot_path)
    print(f"Close Price plot saved at: {close_price_plot_path}")
    plt.close()

    # Plot Trading Volume
    plt.figure(figsize=(12, 6))
    plt.bar(data.index, data['volume'], color='orange')
    plt.title(f"{ticker} - Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.grid()
    volume_plot_path = os.path.join(output_dir, f"{ticker}_volume.png")
    plt.savefig(volume_plot_path)
    print(f"Volume plot saved at: {volume_plot_path}")
    plt.close()

# Example usage
ticker = 'ircon.NS'
stock_data = fetch_data_from_db(ticker)

if stock_data.empty:
    print(f"No data found for {ticker}.")
else:
    plot_stock_data(stock_data, ticker)
