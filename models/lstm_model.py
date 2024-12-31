import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output the predicted stock price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Save the model
model.save('models/lstm_model.h5')
print("LSTM model saved as 'models/lstm_model.h5'")

