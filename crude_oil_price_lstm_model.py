#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_excel("C:/Users/Asus/Documents/oil price prediction_pro/crude_oil_price_daily.xlsx")

# Preprocess data
scaler = MinMaxScaler()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Scaled_Price'] = scaler.fit_transform(df[['Closing Value']])

# Create sequences for LSTM
sequence_length = 30  # Adjust as needed
sequences = []
prices = df['Scaled_Price'].to_numpy()

for i in range(len(prices) - sequence_length):
    sequences.append(prices[i:i+sequence_length+1])

# Convert to NumPy array
sequences = np.array(sequences)

# Split data into features and target
X, y = sequences[:, :-1], sequences[:, -1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Model Loss on Test Data: {loss}")

# Save the model (optional)
model.save("crude_oil_price_lstm_model")


# In[ ]:




