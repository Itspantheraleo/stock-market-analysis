import yfinance as yf
import os

def fetch_stock_data(ticker, start_date, end_date, output_file):
    """
    Fetch historical stock data and save it to a CSV file.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        output_file (str): File path to save the data.
    """
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # Reset index to include 'Date' as a column
        stock_data.reset_index(inplace=True)

        # Debug: Print data to verify correctness
        print("Fetched data preview:")
        print(stock_data.head())

        # Ensure the output folder exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save data to CSV
        stock_data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error fetching data: {e}")

# Example usage
if __name__ == "__main__":
    fetch_stock_data(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2023-12-31",
        output_file="data/AAPL_historical_data.csv"
    )
