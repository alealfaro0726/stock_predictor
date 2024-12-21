import streamlit as st
import yfinance as yf
import json
import pandas as pd
from difflib import get_close_matches
from plotly import graph_objs as go
from datetime import date
import numpy as np
import requests
import torch
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from multiprocessing import freeze_support
from finvizfinance.quote import finvizfinance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, GridSearchCV

import scipy


START = "2020-01-01"
END = date.today().strftime("%Y-%m-%d")

with open('company_tickers.json','r') as f:
    tickers_data = json.load(f)
    
    
def classify_sentiment(title):
    vader = SentimentIntensityAnalyzer()
    
    score = vader.polarity_scores(title)['compound']
    
    return score
    
    
def get_news(stock):
    info = finvizfinance(stock)
    news_df = info.ticker_news()
    
    
    news_df['Title'] = news_df["Title"].str.lower()
    
    news_df['sentiment'] = news_df['Title'].apply(classify_sentiment)
    news_df = news_df[news_df['sentiment'].abs() > 0.01]  
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df['DateOnly'] = news_df['Date'].dt.date
    news_df['ticker'] = stock
    
    mean_df = news_df.groupby('DateOnly')['sentiment'].mean().reset_index()
    # print(mean_df)
    
    return news_df
    

def combine_data(news_df, stock_data):
    stock_data['DateOnly'] = stock_data['Date'].dt.date  # Extract the date part
    combined = pd.merge(stock_data, news_df[['DateOnly', 'sentiment']],
                        left_on='DateOnly',
                        right_on='DateOnly',
                        how="left")
    combined['sentiment'] = combined['sentiment'].fillna(0)
    
    combined['prev_close'] = combined['Close'].shift(1)
    combined['price_change'] = combined['Close'] - combined['prev_close']
    combined['percent_change'] = combined['price_change'] / combined['prev_close']
    combined['volume'] = combined['Volume'].shift(1)
   
    combined.dropna(inplace=True)
    
    # Remove duplicate rows based on the DateOnly column
    combined = combined.drop_duplicates(subset='DateOnly')
    
    print(combined)
    
    return combined

def train_model(comb, future_days):
    dat = comb.filter(['Close'])
    
    dataset = dat.values
    
    training_data_len = int(np.ceil(len(dataset) * 0.95))
    
    print(training_data_len)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    
    for i in range(365, len(train_data)):
        x_train.append(train_data[i-365:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 366:
            print(x_train)
            print(y_train)
            print()
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape = (x_train.shape[1],1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train,y_train, batch_size=1, epochs=1)
    
    test_data = scaled_data[training_data_len - 365: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(365, len(test_data)):
        x_test.append(test_data[i-365:i, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    rmse = np.sqrt(np.mean((predictions-y_test) ** 2))
    
    future_predictions = []
    last_90_days = scaled_data[-30:]
    for _ in range(future_days):
        last_90_days = np.reshape(last_90_days, (1, last_90_days.shape[0],1))
        next_pred = model.predict(last_90_days)[0][0]
        future_predictions.append(next_pred)
        last_90_days = np.append(last_90_days.flatten()[1:], next_pred).reshape(-1,1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

    
    print(rmse)
    
    train = data[:training_data_len]
    valid = data[training_data_len+1:]
    valid['Predictions'] = predictions
    
    future_dates = pd.date_range(start=comb['Date'].iloc[-1], periods=future_days + 1, freq='B')[1:]
    
    
    
    last_temp = comb['Close'].values.reshape(-1,1)
    last_prices = last_temp[-30:]
    last_prices_scaled = scaler.transform(last_prices.reshape(-1, 1))
    
    
    current_input = last_prices_scaled[-30:].reshape(1, 30, 1)
    
    est_pred = []

    for _ in range(0,30):
        predicted_scaled = model.predict(current_input)  # Predict the next step
        est_pred.append(predicted_scaled[0, 0])  # Save the prediction
        
        # Reshape and update current input for the next prediction
        predicted_scaled = predicted_scaled.reshape(1, 1, 1)  # (batch_size, 1, features)
        current_input = np.append(current_input[:, 1:, :], predicted_scaled, axis=1)  # (1, 30, 1)


    predicted_prices = scaler.inverse_transform(np.array(est_pred).reshape(-1, 1))

    print(predicted_prices)

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predictions': future_predictions.flatten()
    })
    
    print(f"future predictions{future_predictions}")
    # print(future_df)
    
    # future_df.set_index('Date', inplace=True)
    
    # combined_df = pd.concat([data, valid[['Predictions']], future_df], axis=0)

    # print(combined_df.tail(future_days + 5))  
    
    
    return future_df, valid;
    

# def train_model(comb):
#     X = comb[['prev_close', 'price_change', 'percent_change', 'sentiment', 'volume']]
#     print(f"Training feature columns: {X.columns}")
    
#     y = comb['Close']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train_scaled, y_train)
    
#     y_pred = model.predict(X_test_scaled)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)

#     return model, scaler
    

# def predict_future(model, scaler, combined_data, days=30):
#     predictions = []
#     last_row = combined_data.iloc[-1]
    
#     for _ in range(days):
#         features = [
#             last_row['prev_close'],
#             last_row['price_change'],
#             last_row['percent_change'],
#             last_row['sentiment'],
#             last_row['volume'],
            
#         ]
        
#         # print(f"Prediction feature columns: {features}")
        
#         # Ensure features are scaled using the same scaler
#         scaled_features = scaler.transform([features])
        
#         next_price = model.predict(scaled_features)[0]    

#         # print(f"Predicted Price: {next_price}")

#         predictions.append(next_price)
#         print(f"Features: {features}, Next Price: {next_price}")

#         last_row = {
#             'prev_close': next_price,
#             'price_change': next_price - last_row['prev_close'],
#             'percent_change': (next_price - last_row['prev_close']) / last_row['prev_close'] if last_row['prev_close'] != 0 else 0,
#             'sentiment': last_row['sentiment'],
#             'volume': last_row['volume'],
            
#         }
        
#     return predictions




    
def generate_future_dates(start_date, num_days):
    return [start_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]

    
def plot_future_data(comb, predictions, future_days):
    st_date = data['Date'].iloc[-1]
    # future_dates = generate_future_dates(last_date, future_days)
    # future_dates = pd.date_range(start=comb['Date'].iloc[-1], periods=future_days, freq='B')[1:]
    # if len(predictions) != len(future_dates):
    #     raise ValueError(f"Mismatch: {len(predictions)} predictions, but {len(future_dates)} future dates.")
    # prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close':predictions})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comb['Date'], y=comb['Close'], name = 'Actual Close Price'))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['Predictions'], name = 'Predicted Close Price',line=dict(dash='dot')))
    fig.update_layout(
        title="Stock Price Prediction",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)
    
    
def plot_pred_data(comb, predictions, future_days):
    st_date = data['Date'].iloc[-1]
    # future_dates = generate_future_dates(last_date, future_days)
    # future_dates = pd.date_range(start=comb['Date'].iloc[-1], periods=future_days, freq='B')[1:]
    # if len(predictions) != len(future_dates):
    #     raise ValueError(f"Mismatch: {len(predictions)} predictions, but {len(future_dates)} future dates.")
    # prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close':predictions})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comb['Date'], y=comb['Close'], name = 'Actual Close Price'))
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['Predictions'], name = 'Predicted Close Price',line=dict(dash='dot')))
    fig.update_layout(
        title="Stock Price Prediction",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)
    


def ticker_exists(stock):
    stock_upper = stock.upper()
    for key, value in tickers_data.items():
        if value['ticker'] == stock_upper:
            return True
    return False
    
def find_similar(input_ticker, num_suggestions=5):
    suggestions = get_close_matches(input_ticker.upper(), [v['ticker'] for k,v in tickers_data.items()], n=num_suggestions, cutoff=0.6)
    
    return suggestions
    
@st.cache_data
def run_prediction(stock):
    data = yf.download(stock, START, END)
    data.reset_index(inplace=True)
    data['Pct_Change'] = data['Close'].pct_change() * 100 
    
    
    
    news = get_news(stock)
    
    # print(data)

    return data



st.title("Stock Prediction")

stock = st.text_input("Stock Ticker: ", placeholder='AAPL, GOOG, MSFT')
stock = stock.upper()
run_btn = st.button("Run Prediction")

n_months = st.slider("Months of prediction", 1, 24)
period = n_months * 30


# def plot_future_date():
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data['Date'],y=data["Open"],name='Stock Open'))
#     fig.add_trace(go.Scatter(x=data['Date'],y=data["Close"],name='Stock Close'))
#     fig.layout.update(title_text="Future Data", xaxis_rangeslider_visible=True)
#     pass;

def plot_raw_data():
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    

if(run_btn):
    stock = stock.strip().upper()
    if ticker_exists(stock):
        data_load_state = st.text("...")
        data = run_prediction(stock)
        data_load_state.text("✓")
        
        
        
        
        
        news_data = get_news(stock)
        
        combined_data = combine_data(news_data, data)
        obser, predictions = train_model(combined_data, period)
        future_days = period
        # pred = predict_future(model, scaler, combined_data)
        
        st.subheader('Raw Data')
        st.write(data.tail())
        
        plot_raw_data()
        
        st.subheader('Predicted Data')
        plot_future_data(data, predictions, future_days)
        
        st.subheader("Future Data")
        plot_pred_data(data, obser, future_days)
        # plot_future_data();
        
    else:
        st.error(f"The stock ticker {stock} does not exist.")
        similar = find_similar(stock)
        if(similar):
            st.info(f"Did you mean {', '.join(similar)}?")
        else:
            st.info("No similar tickers found.")
    

    
    
    
    
#things to do for this project:
# - use sentiment analysis to analyze articles on that particular stock
# - create a plot of previous months/years of the stock's performance
# - create plot of future years or months of the stock
# - create an overall trend and prediction statement/paragraph
# - maybe include a chatbot that users can ask whether the stock will go up or go down in the next years given the data i provide it from this
