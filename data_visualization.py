import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

def visualize_stock_data(file_path):
    """
    Visualizes stock data with price trends, moving averages, and candlestick charts.

    Args:
        file_path (str): Path to the processed stock data file (CSV).
    """
    try:
        # Load processed data
        data = pd.read_csv(file_path, parse_dates=["Date"])
        data.set_index("Date", inplace=True)

        # Plot Closing Prices with Moving Averages
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data["Close"], label="Close Price", color="blue")
        plt.plot(data.index, data["SMA_50"], label="50-Day SMA", color="orange")
        plt.plot(data.index, data["EMA_50"], label="50-Day EMA", color="green")
        plt.title("Stock Price with Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.show()

        # Candlestick Chart
        ohlc_data = data[["Open", "High", "Low", "Close"]].copy()
        ohlc_data.reset_index(inplace=True)
        ohlc_data["Date"] = ohlc_data["Date"].apply(mdates.date2num)

        fig, ax = plt.subplots(figsize=(14, 7))
        candlestick_ohlc(ax, ohlc_data.values, width=0.6, colorup="green", colordown="red", alpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_title("Candlestick Chart")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.grid()
        plt.show()

        # Volume Analysis
        plt.figure(figsize=(14, 5))
        plt.bar(data.index, data["Volume"], color="gray", alpha=0.7)
        plt.title("Trading Volume Over Time")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"Error during visualization: {e}")

# Example usage
if __name__ == "__main__":
    visualize_stock_data("data/processed/AAPL_processed_data.csv")
