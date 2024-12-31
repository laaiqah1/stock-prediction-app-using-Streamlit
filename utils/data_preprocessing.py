import yfinance as yf
import pandas as pd
import numpy as np

# Fetch historical stock data
def fetch_data(stock_symbol, start_date, end_date):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    return df

# Manual calculation of technical indicators
def add_technical_indicators(df):
    # 1. Simple Moving Average (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 2. Relative Strength Index (RSI)
    delta = df['Close'].diff()  # Price difference between consecutive days
    gain = delta.where(delta > 0, 0)  # Gains, i.e., positive differences
    loss = -delta.where(delta < 0, 0)  # Losses, i.e., negative differences

    avg_gain = gain.rolling(window=14).mean()  # Rolling average gain
    avg_loss = loss.rolling(window=14).mean()  # Rolling average loss

    rs = avg_gain / avg_loss  # Relative Strength (RS)
    df['RSI'] = 100 - (100 / (1 + rs))  # RSI formula

    # 3. Moving Average Convergence Divergence (MACD)
    # MACD is the difference between the 12-day and 26-day EMAs
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = df['EMA_12'] - df['EMA_26']  # MACD Line
    df['Signal_line'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line

    return df

