# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from alpha_vantage.timeseries import TimeSeries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Fetch historical stock data using Alpha Vantage API
API_KEY = 'E7AQQJIUBPTIP6W2'  # Replace with your valid API key
SYMBOL = 'MSFT'
timeseries = TimeSeries(key=API_KEY, output_format='pandas')
data, meta_data = timeseries.get_daily(symbol=SYMBOL, outputsize='full')

# Data preparation
data = data.sort_index()  # Ensure the data is sorted by date
data_close = data['4. close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close)

# Create dataset for LSTM model
look_back = 60

def create_dataset(dataset, look_back=look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X, Y = create_dataset(scaled_data)

# Split into train and test sets
train_size = int(len(X) * 0.80)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Create DataFrame for actual and predicted stock prices
actual_stock_price = data_close[train_size + look_back:]
predicted_stock_price = predicted_stock_price[:len(actual_stock_price)]

# Visualize the results
plt.figure(figsize=(16,8))
plt.plot(actual_stock_price, color='blue', label='Actual MSFT Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted MSFT Stock Price')
plt.title(f'{SYMBOL} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(actual_stock_price, predicted_stock_price)
print(f'Mean Squared Error: {mse}')
