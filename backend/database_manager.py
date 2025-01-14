import sqlite3

def create_database():
    conn = sqlite3.connect('stock_data4.db')
    cursor = conn.cursor()
    
    # Create table
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

def save_to_database(ticker, data):
    conn = sqlite3.connect('stock_data4.db')
    cursor = conn.cursor()
    
    for date, row in data.iterrows():
        # Convert pandas Timestamp to string
        date_str = date.strftime('%Y-%m-%d')  # Convert to 'YYYY-MM-DD' format
        
        cursor.execute('''
            INSERT INTO stock_prices (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, date_str, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
    
    conn.commit()
    conn.close()
    print(f"Data for {ticker} saved successfully.")

# Example Usage
if __name__ == "__main__":
    from data_fetcher import fetch_stock_data

    create_database()
    ticker = "ircon.NS"
    stock_data = fetch_stock_data(ticker, "2024-01-01", "2024-12-31")
    if stock_data is not None:
        save_to_database(ticker, stock_data)
