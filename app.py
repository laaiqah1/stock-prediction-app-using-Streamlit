import streamlit as st
import pandas as pd
from models.lstm_model import build_lstm_model
from models.cnn_model import build_cnn_model
from models.rnn_model import build_rnn_model
from models.xgboost_model import train_xgboost_model
from utils.data_preprocessing import fetch_data, add_technical_indicators
import plotly.graph_objects as go
import numpy as np

st.title('Stock Prediction App')

# Input for stock symbol and date range
stock_symbol = st.text_input('Enter Stock Symbol', 'AAPL')
start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2023-01-01'))

# Fetch stock data and technical indicators
df = fetch_data(stock_symbol, start_date, end_date)
df = add_technical_indicators(df)

# Plot stock data and indicators
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlesticks'))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='50-Day SMA'))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='200-Day SMA'))
fig.update_layout(title=f'{stock_symbol} Stock Data', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)

# Placeholder for the model selection
model_choice = st.selectbox('Select Prediction Model', ['LSTM', 'CNN', 'RNN', 'XGBoost'])

# Preprocessing and prediction handling
def prepare_data_for_model(df):
    # Prepare data for model input
    features = df[['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD']].values
    target = df['Close'].shift(-1).dropna().values
    features = features[:-1]
    return features, target

features, target = prepare_data_for_model(df)

# Handle the model prediction based on selected model
if model_choice == 'LSTM':
    st.write("Running LSTM Model...")
    model = build_lstm_model(input_shape=(features.shape[1], 1))
    # Assuming the model is pre-trained and saved as `lstm_model.h5`
    model.load_weights('models/lstm_model.py')
    predictions = model.predict(features)
    st.write(predictions)

elif model_choice == 'CNN':
    st.write("Running CNN Model...")
    model = build_cnn_model(input_shape=(features.shape[1], 1))
    model.load_weights('models/cnn_model.py')
    predictions = model.predict(features)
    st.write(predictions)

elif model_choice == 'RNN':
    st.write("Running RNN Model...")
    model = build_rnn_model(input_shape=(features.shape[1], 1))
    model.load_weights('models/rnn_model.py')
    predictions = model.predict(features)
    st.write(predictions)

elif model_choice == 'XGBoost':
    st.write("Running XGBoost Model...")
    model = train_xgboost_model(features, target)
    predictions = model.predict(features)
    st.write(predictions)
