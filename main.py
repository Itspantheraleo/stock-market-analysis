from data_collection import get_stock_data

def main():
    # Take user inputs
    ticker = input("Enter the stock ticker (e.g., AAPL): ").strip()
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

    # Fetch stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    if stock_data is not None:
        print("Stock data fetched successfully!")
        print(stock_data.head())
    else:
        print("Failed to fetch stock data.")

if __name__ == "__main__":
    main()
