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
        # Ensure file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        # Load data
        data = pd.read_csv(file_path)

        # Debugging: Print the first few rows and data types
        print("Data preview before cleaning:")
        print(data.head())
        print("\nData types before cleaning:")
        print(data.dtypes)

        # Check for required columns
        required_columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing one or more required columns: {required_columns}")

        # Drop rows where "Date" is NaN (likely caused by an invalid header row)
        data = data.dropna(subset=["Date"])

        # Parse dates and handle inconsistent formats
        data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
        data = data.dropna(subset=["Date"])  # Drop rows with invalid dates

        # Ensure numeric columns are properly cast
        numeric_columns = ["Close", "High", "Low", "Open", "Volume"]
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Drop rows where "Close" or other key columns are still NaN
        data = data.dropna(subset=["Close", "High", "Low", "Open", "Volume"])

        # Sort data by Date
        data.sort_values("Date", inplace=True)

        # Debugging: Print cleaned data
        print("Data preview after cleaning:")
        print(data.head())
        print("\nData types after cleaning:")
        print(data.dtypes)

        # Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
        sma_windows = [20, 50, 100]
        ema_windows = [20, 50, 100]

        for window in sma_windows:
            data[f"SMA_{window}"] = data["Close"].rolling(window=window).mean()

        for window in ema_windows:
            data[f"EMA_{window}"] = data["Close"].ewm(span=window, adjust=False).mean()

        # Add additional debug information
        print("\nData preview with moving averages:")
        print(data.head(10))  # Print first 10 rows for better debugging

        # Ensure the processed data folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save processed data
        data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
    except ValueError as val_error:
        print(f"Value error: {val_error}")
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")

# Example usage
if __name__ == "__main__":
    preprocess_stock_data(
        "data/AAPL_historical_data.csv",
        "data/processed/AAPL_processed_data.csv"
    )
