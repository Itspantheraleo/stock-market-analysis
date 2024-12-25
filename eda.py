import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path):
    """
    Perform exploratory data analysis on the processed stock data.

    Args:
        file_path (str): Path to the processed stock data CSV.
    """
    try:
        # Load the processed data
        data = pd.read_csv(file_path)

        # Convert 'Date' to datetime format
        data['Date'] = pd.to_datetime(data['Date'])

        # Basic statistics
        print("Data Overview:")
        print(data.describe())

        # Plot Close price over time
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Close'], label="Close Price", color='blue')
        plt.plot(data['Date'], data['SMA_50'], label="SMA_50", color='orange')
        plt.plot(data['Date'], data['EMA_50'], label="EMA_50", color='green')
        plt.title("Stock Price and Moving Averages Over Time")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        # Plot volume over time
        plt.figure(figsize=(12, 6))
        plt.bar(data['Date'], data['Volume'], color='purple', alpha=0.5)
        plt.title("Trading Volume Over Time")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    except Exception as e:
        print(f"Error during EDA: {e}")

# Example usage
if __name__ == "__main__":
    perform_eda("data/processed/AAPL_processed_data.csv")
