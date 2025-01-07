import sqlite3
from webbrowser import get
import yfinance as yf
import time
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred

FRED_API_KEY = "5210e9772c9a026e438e52c4f82a3edc"
fred = Fred(api_key=FRED_API_KEY)
def del_database():
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        DROP TABLE stocks
    ''')
    conn.commit()
    conn.close()


def create_database():
    """Creates a SQLite database and the necessary table for stock data."""
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            ticker TEXT,
            date TEXT,
            closing_price REAL,
            average_volume REAL,
            market_cap REAL,
            mma_30 REAL,
            mma_90 REAL,
            PRIMARY KEY (ticker, date)
        )
    ''')

    # Table for macroeconomic data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS macroeconomic_data (
            indicator TEXT,
            date TEXT,
            value REAL,
            PRIMARY KEY (indicator, date)
        )
    ''')

    conn.commit()
    conn.close()

def fetch_macro_data(indicator, start_date, end_date):
    """Fetches macroeconomic data from the FRED API."""
    try:
        data = fred.get_series(indicator, start_date, end_date)
        return data
    except Exception as e:
        print(f"Error fetching data for {indicator}: {e}")
        return None

def store_macro_data(indicator, data):
    """Stores macroeconomic data in the database."""
    if data is None or data.empty:
        print(f"No data to store for {indicator}.")
        return

    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()

    # Prepare data for insertion
    rows = [(indicator, date.strftime('%Y-%m-%d'), value) for date, value in data.items()]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO macroeconomic_data (indicator, date, value)
        VALUES (?, ?, ?)
    ''', rows)
    conn.commit()
    conn.close()
    print(f"Stored data for {indicator}.")

def get_latest_macro_data(macro):
    """Fetches the most recent macro data for the given ticker and prints it on one line."""
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()

    # Query to get the latest stock data
    cursor.execute('''
        SELECT date, indicator, value 
        FROM macroeconomic_data 
        WHERE indicator = ? 
        ORDER BY date DESC 
        LIMIT 1
    ''', (macro,))
    
    result = cursor.fetchone()
    conn.close()

    if not result:
        print(f"No data found for ticker {macro}.")
        return

    # Extract the most recent data
    latest_date, indicator_name, value = result

    # Print data on one line
    print(f"{latest_date:<12} {indicator_name:<15} {value:<15.2f}")

def update_macro_data(indicators, start_date=None):
    """Updates macroeconomic data for the given indicators."""
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    for indicator in indicators:
        print(f"Updating data for {indicator}...")
        data = fetch_macro_data(indicator, start_date, end_date)
        store_macro_data(indicator, data)

def fetch_and_store_data(ticker, start_date, end_date):
    """Fetches stock data from yfinance, calculates MMAs, and stores it in the database."""
    stock = yf.Ticker(ticker)
    history = stock.history(start=start_date, end=end_date)
   
    if history.empty:
        print(f"No data found for {ticker}.")
        return

    market_cap = 0
    shares = stock.info.get('sharesOutstanding', 0)
    # Calculate moving averages
    history['mma_30'] = history['Close'].rolling(window=30).mean()
    history['mma_90'] = history['Close'].rolling(window=90).mean()

    # Manually calculate the last row's moving averages if needed
    if len(history) >= 30:
        last_30 = history['Close'][-30:].mean()
    else:
        last_30 = history['Close'].mean()

    if len(history) >= 90:
        last_90 = history['Close'][-90:].mean()
    else:
        last_90 = history['Close'].mean()

    # Assign manual values to the last row
    if pd.isna(history['mma_30'].iloc[-1]):
        history.loc[history.index[-1], 'mma_30'] = last_30
    if pd.isna(history['mma_90'].iloc[-1]):
        history.loc[history.index[-1], 'mma_90'] = last_90
    data = []
    #print(history.tail(2))
    for date, row in history.iterrows():
        market_cap = shares * row['Close']
        mma_30 = row['mma_30'] if pd.notna(row['mma_30']) else row['Close']
        mma_90 = row['mma_90'] if pd.notna(row['mma_90']) else row['Close']
        data.append((
            ticker,
            date.strftime('%Y-%m-%d'),
            row['Close'],
            row['Volume'],
            market_cap,
            mma_30,
            mma_90
        ))

    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT OR REPLACE INTO stocks (
            ticker, date, closing_price, average_volume, market_cap, mma_30, mma_90
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()
    print(data[len(data)-1:])

def fetch_and_store_data_real_time(ticker):
    """Fetches the most recent stock data, calculates MMAs, and updates the database."""
    stock = yf.Ticker(ticker)
    # Fetch the last 90 days of data to compute moving averages
    history = stock.history(period="3mo")

    if history.empty:
        print(f"No data found for {ticker}.")
        return

    # Calculate moving averages
    if len(history) >= 30:
        last_30 = history['Close'][-30:].mean()
    else:
        last_30 = history['Close'].mean()

    if len(history) >= 90:
        last_90 = history['Close'][-90:].mean()
    else:
        last_90 = history['Close'].mean()
    # Get the most recent date
    latest_row = history.iloc[-1]
    latest_date = history.index[-1]


    closing_price = latest_row['Close']
    volume = latest_row['Volume']
    market_cap = closing_price * stock.info.get('sharesOutstanding', 0)

    # Prepare the data for insertion
    data = (
        ticker,
        latest_date.strftime('%Y-%m-%d'),
        closing_price,
        volume,
        market_cap,
        last_30,
        last_90
    )

    # Connect to the database and update or insert the data
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO stocks (
            ticker, date, closing_price, average_volume, market_cap, mma_30, mma_90
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

    print(f"Updated stock data for {ticker}")
    

def update_data(ticker):
    """Updates the stock data for a specific ticker."""
    fetch_and_store_data_real_time(ticker)

def query_data(ticker, start_date=None, end_date=None):
    """Queries the database for a specific ticker and time window."""
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    if start_date is None:
        start_date = '2024-01-01'  # Default to earliest possible date

    cursor.execute('''
        SELECT date, closing_price, average_volume, market_cap , mma_30, mma_90
        FROM stocks 
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date ASC
    ''', (ticker, start_date, end_date))
    
    results = cursor.fetchall()
    conn.close()

    if not results:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return

    print(f"Data for {ticker} from {start_date} to {end_date}:")
    print(f"{'Date':<12} {'Closing Price':<15} {'Average Volume':<15} {'Market Cap':<15}   {'MMA_30':<15} {'MMA_90':<15}")
    print("-" * 90)
    for row in results:
        print(f"{row[0]:<12} {row[1]:<15.2f} {row[2]:<15.2f} {row[3]:<15.2f}     {row[4]:<15.2f} {row[5]:<15.2f}")

def get_latest_stock_data(ticker):
    """Fetches the most recent stock data for the given ticker and prints it on one line."""
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()

    # Query to get the latest stock data
    cursor.execute('''
        SELECT date, ticker, closing_price 
        FROM stocks 
        WHERE ticker = ? 
        ORDER BY date DESC 
        LIMIT 1
    ''', (ticker,))
    
    result = cursor.fetchone()
    conn.close()

    if not result:
        print(f"No data found for ticker {ticker}.")
        return

    # Extract the most recent data
    latest_date, stock_name, closing_price = result

    # Print data on one line
    print(f"{latest_date:<12} {stock_name:<15} {closing_price:<15.2f}")

def get_current_stock_info(ticker_symbol):
    """Fetches and prints the current stock information for the given ticker."""
    try:
        # Create a Ticker object
        ticker = yf.Ticker(ticker_symbol)

        # Fetch live stock history
        stock_data = ticker.history(period="1d")  # Fetch today's data
        
        # Ensure there is data
        if stock_data.empty:
            print(f"No data available for ticker {ticker_symbol}.")
            return

        # Extract the current price
        current_price = stock_data['Close'].iloc[-1]

        # Print the stock information
        print(f"Ticker: {ticker_symbol.upper()}")
        print(f"Current Price: {current_price:.2f}")

    except Exception as e:
        print(f"Error fetching stock information: {e}")
# Main script
if __name__ == "__main__":
    #del_database()
    #create_database()
    macro_indicators = ['UNRATE', 'CPIAUCSL', 'VIXCLS']  # Unemployment, Inflation, Volatility Index
    # Fetch data for multiple stocks
    stocks = ['AAPL', 'MSFT', 'GOOGL']  # Example tickers
    start_date = '2022-01-07'
    end_date = '2025-01-7'
    fetch_and_store_data('AAPL', start_date, end_date)
    update_data('AAPL')
    query_data('AAPL', start_date='2024-11-01')
    #for data in macro_indicators:
        #fetch_macro_data(data,start_date,end_date)
    """for stock in stocks:
        fetch_and_store_data(stock, start_date, end_date)
        update_data(stock)
        query_data(stock, start_date='2024-11-01')"""
    #get_current_stock_info("AAPL")
    #print("Stock data has been successfully stored in the database.")
    #update_macro_data(macro_indicators,"2024-1-1")
    """while True:
        for stock in stocks:
            update_data(stock)
        print("")
        for stock in stocks:
            get_latest_stock_data(stock)
        print("")
        for macro in macro_indicators:
            get_latest_macro_data(macro)
        print("")
        time.sleep(10)"""



