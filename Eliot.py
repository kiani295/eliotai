
from __future__ import annotations
from flask import Flask, render_template, request, jsonify
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern, convert_yf_data
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from stock import Stock
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources if not already downloaded
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# تنظیمات منطقی
MAP_CVD = 200  # Moving Average CVD Period

MAP_CVD = 200  # Moving Average CVD Period
MAP_Vol = 100  # Moving Average Volume Period
BarBack = 10  # Level Finder Bar Back
Refinder = 5  # Levels update per candles
ATR_On = True  # ATR On
ATR_Multi = 1.0  # ATR Multiplier

# بازارها
market_data = {
    'Forex': ['FX', 'OANDA', 'FOREXCOM', 'FX_IDC', 'PEPPERSTONE', 'CAPITALCOM', 'ICMARKETS', 'EIGHTCAP', 'SAXO', 'BLACKBULL', 'VANTAGE', 'FUSIONMARKETS', 'FPMARKETS', 'GBEBROKERS', 'IBKR', 'ACTIVTRADES', 'EASYMARKETS', 'FXOPEN', 'CITYINDEX', 'AFTERPRIME', 'SKILLING', 'WHSELFINVEST', 'TRADENATION', 'THINKMARKETS', 'CFI', 'PHILLIPNOVA'],
    'Crypto': ['BITSTAMP', 'COINBASE', 'INDEX', 'CRYPTO', 'BINANCE', 'BITFINEX', 'KRAKEN', 'OANDA', 'PEPPERSTONE', 'GEMINI', 'EIGHTCAP', 'ICMARKETS', 'VANTAGE', 'CAPITALCOM', 'FOREXCOM', 'FX', 'BLACKBULL', 'SAXO', 'FUSIONMARKETS', 'CRYPTOCOM', 'EASYMARKETS', 'OKCOIN', 'FPMARKETS', 'AFTERPRIME', 'ACTIVTRADES', 'BTSE'],
    'Stock': ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
}

def find_start_of_wave_three(df):
    wave_analyzer = WaveAnalyzer(df)
    wave_starts = []
    
    if wave_analyzer:
        wave_starts.append(df)
    
    return wave_starts

def save_chart(currencies):
    plt.figure(figsize=(10, 5))
    
    for ticker, df in currencies.items():
        dates = df.index
        prices = df['Close']
        
        plt.plot(dates, prices, label=ticker)
        
    plt.title('Start of Wave Three')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('wave_three_chart.png')
    plt.close()

def get_historical_data_from_nobitex(symbol):
    url = f"https://api.nobitex.ir/v3/orderbook/all"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching data for {symbol}: {response.status_code} - {response.text}")
        return pd.DataFrame()
    
    data = response.json()
    
    if 'status' not in data or data['status'] != 'ok':
        print(f"No price data found for {symbol}. Response: {data}")
        return pd.DataFrame()
    
    historical_data = []
    if symbol in data:
        last_update = datetime.fromtimestamp(data[symbol]['lastUpdate'] / 1000)
        open_price = float(data[symbol]['bids'][0][0]) if data[symbol]['bids'] else None
        high_price = max(float(entry[0]) for entry in data[symbol]['asks']) if data[symbol]['asks'] else None
        low_price = min(float(entry[0]) for entry in data[symbol]['bids']) if data[symbol]['bids'] else None
        
        for entry in data[symbol]['bids']:
            price = float(entry[0])
            volume = float(entry[1])
            historical_data.append([last_update, open_price, high_price, low_price, price])
    
    # Create DataFrame and set index
    df = pd.DataFrame(historical_data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
    df.set_index('Date', inplace=True)  # Set 'Date' as index
    return df

def get_crypto_tickers():
    url = "https://api.nobitex.ir/v3/orderbook/all"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching tickers: {response.status_code} - {response.text}")
        return []
    
    data = response.json()
    
    if 'status' not in data or data['status'] != 'ok':
        print("No data found in the response.")
        return []
    
    tickers = [key for key in data.keys() if key != 'status']
    return tickers

def analyze_data(result_text, days, hours):
    historical_data = {}
    tickers = get_crypto_tickers()
    filtered_currencies = {}
    potential_buy_currencies = []

    for ticker in tickers:
        df = get_historical_data_from_nobitex(ticker)
        if not df.empty:
            historical_data[ticker] = df

    start_wave_three_currencies = {}

    for ticker, df in historical_data.items():
        stock = Stock(ticker, df.index.min(), df.index.max())
        
        if stock.rsi is not None and len(stock.rsi) > 0:
            processed_data = convert_yf_data(df)
            if processed_data is not None and not processed_data.empty:
                wave_starts = find_start_of_wave_three(processed_data)
                if wave_starts:
                    start_wave_three_currencies[ticker] = wave_starts[0]
                    current_rsi = stock.rsi[-1]
                    if 10 < current_rsi < 20:
                        filtered_currencies[ticker] = current_rsi

                        # بررسی CVD برای پتانسیل خرید
                        V = df['Close'].values
                        delta = np.diff(V)
                        CVD = np.cumsum(delta)
                        CVD_Avg = pd.Series(CVD).ewm(span=MAP_CVD).mean().to_numpy()

                        # بررسی روند صعودی CVD
                        if CVD[-1] > CVD_Avg[-1]:
                            potential_buy_currencies.append(ticker)

            else:
                print(f"Processed data for {ticker} is None or empty.")

    if start_wave_three_currencies:
        save_chart(start_wave_three_currencies)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "\n".join([f"- {ticker} (RSI: {filtered_currencies.get(ticker, 'N/A')})" for ticker in start_wave_three_currencies.keys()]))
        
        # نمایش ارزهای با پتانسیل خرید
        if potential_buy_currencies:
            result_text.insert(tk.END, "\n\nPotential Buy Currencies based on CVD:\n")
            result_text.insert(tk.END, "\n".join(potential_buy_currencies))
        else:
            result_text.insert(tk.END, "\n\nNo currencies found with potential for buying based on CVD.")
    else:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "هیچ ارزی در ابتدای موج سوم شناسایی نشد.")

    # Add machine learning features
    for ticker, df in historical_data.items():
        # Price Prediction using Linear Regression
        df['Date'] = df.index.map(datetime.toordinal)
        X = df[['Date']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_prices = model.predict(X_test)
        
        # Clustering using KMeans
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])
        kmeans = KMeans(n_clusters=3)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        # Sentiment Analysis (Placeholder)
        sentiment_score = sia.polarity_scores("Sample news about " + ticker)['compound']
        
        # Display results
        print(f"{ticker} - Predicted Prices: {predicted_prices}, Sentiment Score: {sentiment_score}")

def convert_yf_data(df):
    if df.empty:
        return None
    
    processed_data = pd.DataFrame({
        'Date': df.index,
        'Open': df['Open'],
        'High': df['High'],
        'Low': df['Low'],
        'Close': df['Close']
    })
    
    return processed_data.set_index('Date')

class WaveAnalyzer:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        pass

# محاسبات CVD و تحلیل داده‌های بازار
def calculate_cvd_and_plot():
    historical_data = {}
    tickers = get_crypto_tickers()

    for ticker in tickers:
        df = get_historical_data_from_nobitex(ticker)
        if not df.empty:
            historical_data[ticker] = df

    for ticker, df in historical_data.items():
        V = df['Close'].values
        delta = np.diff(V)
        CVD = np.cumsum(delta)
        CVD_Avg = pd.Series(CVD).ewm(span=MAP_CVD).mean().to_numpy()

        plt.figure(figsize=(12, 6))
        plt.plot(CVD, label='Cumulative Volume Delta', color='blue')
        plt.plot(CVD_Avg, label='CVD Average', color='orange')
        plt.title(f'Cumulative Volume Delta for {ticker}')
        plt.xlabel('Time')
        plt.ylabel('CVD Value')
        plt.legend()
        plt.show()

# تابع اصلی

# توابع شما
# (کدهای شما از جمله توابع get_historical_data_from_nobitex، analyze_data و غیره در اینجا قرار می‌گیرند)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    start_date = request.form['start_date']
    days = int(request.form['days'])
    hours = int(request.form['hours'])
    
    # اینجا می‌توانید تابع analyze_data را فراخوانی کنید
    result_text = analyze_data(start_date, days, hours)
    
    return jsonify(result=result_text)

if __name__ == "__main__":
    app.run(debug=True)