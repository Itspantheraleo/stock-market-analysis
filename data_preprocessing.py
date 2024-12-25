import pandas as pd
import os

def preprocess_stock_data(file_path, output_path):
    """
    Preprocesses stock data by cleaning it and adding moving averages.

    Args:
        file_path (str): Path to the raw stock data CSV.
        output_path (str): Path to save the processed data.
    """
    try:
        # Load data
        data = pd.read_csv(file_path)

        # Debugging: Print the first few rows and data types
        print("Data preview before cleaning:")
        print(data.head())
        print("\nData types before cleaning:")
        print(data.dtypes)

        # Drop rows where "Date" is NaN (likely caused by an invalid header row)
        data = data.dropna(subset=["Date"])

        # Ensure numeric columns are properly cast
        numeric_columns = ["Close", "High", "Low", "Open", "Volume"]
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Drop rows where "Close" is still NaN (invalid rows)
        data = data.dropna(subset=["Close"])

        # Parse dates and sort by Date
        data["Date"] = pd.to_datetime(data["Date"])
        data.sort_values("Date", inplace=True)

        # Debugging: Print cleaned data
        print("Data preview after cleaning:")
        print(data.head())
        print("\nData types after cleaning:")
        print(data.dtypes)

        # Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

        # Ensure the processed data folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save processed data
        data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

# Example usage
if __name__ == "__main__":
    preprocess_stock_data(
        "data/AAPL_historical_data.csv",
        "data/processed/AAPL_processed_data.csv"
    )
