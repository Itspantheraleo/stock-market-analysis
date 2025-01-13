import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker symbol.
    Args:
        ticker (str): Stock symbol with exchange suffix (e.g., 'RELIANCE.NS').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: Historical stock data with Open, High, Low, Close, and Volume.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print("No data found for the given ticker and dates.")
            return None
        else:
            print(f"Data fetched successfully for {ticker}")
            return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    # Example: Fetching data for Reliance Industries (NSE)
    ticker = "ircon.NS"  # NSE Stock
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        print(stock_data.head())
