#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the LSTM model
def load_lstm_model():
    model = load_model('C:/Users/Asus/Documents/oil price prediction_pro1/crude_oil_price_lstm_model')  
    return model

# Function to preprocess data for LSTM model
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Scaled_Price'] = scaler.fit_transform(df[['Closing Value']])
    return df, scaler

# Function to predict future data using the LSTM model
def predict_future_data(model, scaler, input_data):
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction

# Load the dataset
df = pd.read_excel('C:/Users/Asus/Documents/oil price prediction_pro1/crude_oil_price_daily.xlsx') 
df = df.dropna()
df.isnull().sum()

# Preprocess data
processed_data, scaler = preprocess_data(df)

# Streamlit app
def main():
    st.title("Oil Price Prediction")

    # User input for date selection
    selected_month = st.selectbox("Select a month:", range(1, 13))
    selected_year = st.number_input("Select a year:", min_value=processed_data.index.year.min(), max_value=processed_data.index.year.max())

    # User input for prediction
    if st.button("Predict"):
        # Convert user input to datetime
        selected_date = pd.to_datetime(f"{selected_year}-{selected_month}-01")
    
        # Get the previous 30 days of data, if available
        input_data = processed_data.loc[:selected_date]['Scaled_Price'].tail(30)
    
        # Check if there are enough data points
        if len(input_data) >= 30:
            # Reshape input data for LSTM model
            input_data = np.reshape(input_data.values, (1, 30, 1))
        
            # Load the LSTM model
            lstm_model = load_lstm_model()
        
            # Predict for the selected date
            prediction = predict_future_data(lstm_model, scaler, input_data)
        
            # Display the prediction
            st.write(f"Prediction for {selected_date}: ${prediction[0, 0]:.2f}")
        else:
            st.warning("Insufficient data for prediction. Please select an earlier date.")

if __name__ == "__main__":
    main()



# In[ ]:




