
# run the following command on terminal

# streamlit run C:\Users\Senju\Documents\_Solent\24_25_MSc_AI_DataScience\2nd_sem\AI_in_Business\assessment\AI_in_Business_AE2_Pycharm\GUI.py

import streamlit as st
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from xgboost import XGBRegressor
from math import sqrt
from prophet import Prophet
import xgboost as xgb
import pmdarima as pm
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

import os
os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(layout="wide")

# === Load Dataset ===
@st.cache_data
def data_collection(up_to_date=False):
    # The top 30 cryptocurrencies on yahoo finance fluctuates, therefore using an API to fetch the up-to-date data is vital
    # The following data would be those at the time of the assessment (from 2024-05-08 to 2025-05-07)

    # Get top 50 cryptocurrencies from CoinGecko
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 50,  # Get the top 50 - including backups in case a currency contains NaN values as data
        "page": 1,
        "sparkline": False
    }

    response = requests.get(url, params=params)
    top_coins = response.json()

    # Convert symbols to Yahoo Finance tickers (Most work as "SYMBOL-USD")
    crypto_tickers = [coin["symbol"].upper() + "-USD" for coin in top_coins]

    # Set the start date and end date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(365-1))
    # 365 days to indicate one year

    if up_to_date:
        df = yf.download(crypto_tickers, start=start_date, end=end_date)
    else:
        df = yf.download(crypto_tickers, start='2024-05-08',end='2025-05-08')
    # CAUTION! -- downloading from yfinance results in a multi-index dataframe

    # Grouping the cryptocurrencies
    df = df.Close
    # Close = the price when the market closed in the afternoon

    # reset index - separates index (named 'Date') and 'Ticker' column
    # df = df.reset_index()

    # Check if the dataframe contain any NaN values
    if df.isna().any().any() == True:
        # Check which columns contain any NaN values
        nan_columns = df.columns[df.isna().any()]

        # Drop the currencies with NaN values
        df.drop(columns=nan_columns, inplace=True)

        # Truncate the dataframe if bigger than 30 columns
        if df.shape[1] > 30:
            df = df.iloc[:, :30]

    return df

def k_means_clustering(df):
    # Transpose
    df_t = df.T

    # Min-max scaling

    # Transpose so that scaling is applied per coin (row)
    scaled_df = pd.DataFrame(
        MinMaxScaler().fit_transform(df_t),
        index=df_t.index,
        columns=df_t.columns
    )

    # Autoencoder - as dimensionality reduction technique because non-linearity was detected

    # Convert to NumPy array
    X = scaled_df.values.astype(np.float32)

    encoding_dim = 4  # reduce to 4 features - spits error if smaller than number of clusters e.g., 2

    # Input layer
    input_layer = Input(shape=(X.shape[1],))

    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded_output = Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = Dense(32, activation='relu')(encoded_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded_output = Dense(X.shape[1], activation='sigmoid')(decoded)

    # Autoencoder model
    autoencoder = Model(input_layer, decoded_output)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model
    autoencoder.fit(X, X, epochs=100, batch_size=8, shuffle=True, verbose=1)

    # Define the encoder model (for extracting the compressed representation)
    encoder = Model(input_layer, encoded_output)

    # Get compressed features
    encoded_X = encoder.predict(X)

    # Convert to DataFrame
    encoded_df = pd.DataFrame(encoded_X, index=scaled_df.index)

    # K-Means Clustering

    # Ensure all column names are strings
    encoded_df.columns = encoded_df.columns.astype(str)

    # Define the model with recommended n_init fix
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)  # number of clusters = 4

    # Fit and predict cluster labels
    clusters = kmeans.fit_predict(encoded_df)

    # Visualization

    # Add cluster labels to the DataFrame
    encoded_df['Cluster'] = clusters

    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(encoded_df.drop('Cluster', axis=1))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=encoded_df['Cluster'], cmap='viridis')
    ax.set_title('K-Means Clusters of Top 30 Cryptocurrencies\n(After Min-Max Scaling and Autoencoder)')
    cbar = plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(True)

    cluster_groups = {
        cluster: encoded_df[encoded_df['Cluster'] == cluster].index.to_numpy()
        for cluster in encoded_df['Cluster'].unique()
    }

    return fig, cluster_groups

def top_correlated(df, currency, top_n, plot, cmap):
    # Drop any columns with NaNs
    df = df.dropna(axis=1)

    # Ensure target coin is in the DataFrame
    if currency not in df.columns:
        raise ValueError(f"{currency} not found in the DataFrame.")

    # Compute correlation matrix
    corr_matrix = df.corr(method='pearson')

    # Get correlation values for the target coin (excluding itself)
    coin_corr = corr_matrix[currency].drop(labels=[currency])

    # Sort by absolute correlation and return top N
    top_corr = coin_corr.reindex(coin_corr.abs().sort_values(ascending=False).index).head(top_n)

    if plot == True:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(top_corr.to_frame().T, annot=True, cmap=cmap, center=0, cbar=True, ax=ax)
        ax.set_title(f'Top {top_n} Correlated Cryptocurrencies with {currency}')
        ax.set_yticks([])  # Hide y-axis since it's only one row
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()

        return fig

def load_lstm(df, ticker, seq_len=60):
    model_path = f"models/LSTM/{ticker.lower().replace('-', '_')}_lstm_model.keras"

    history_path = f"models/LSTM/{ticker}_lstm_history.pkl"

    scaler_path = f'models/LSTM/{ticker}_lstm_scaler.pkl'

    # Load model WITHOUT loading optimizer state
    model = load_model(model_path, compile=False)

    # Recompile manually to avoid optimizer loading issues
    model.compile(optimizer=Adam(learning_rate=0.003), loss='mse')

    history = None
    if history_path:
        try:
            history = joblib.load(history_path)
        except FileNotFoundError:
            print("History file not found. Returning model without history.")

    scaler = joblib.load(scaler_path)

    prices = df[ticker].values.reshape(-1, 1)

    scaled = scaler.transform(prices)

    # Create sequences
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, seq_len)

    # Split data
    total = len(X)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Inverse transform predictions and true values
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test)

    # Evaluation metrics
    rmse = np.sqrt(np.mean((y_pred - y_actual) ** 2))
    mae = np.mean(np.abs(y_pred - y_actual))
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

    msg = (f"{ticker} Evaluation Metrics (LSTM)\n\n"
           f"- Test RMSE (USD): {rmse:.2f}\n\n"
           f"- Test MAE  (USD): {mae:.2f}\n\n"
           f"- Test MAPE (%): {mape:.2f}")

    # Inverse transform to get original scale
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    # Plot training and validation loss
    fig1, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['loss'], label='Training Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.set_title(f'{ticker} - Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    ax.grid(True)
    fig1.tight_layout()

    # Plot predicted vs actual values
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_actual, label='Actual', color='black')
    ax.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    ax.set_title(f"{ticker} - Actual vs Predicted Prices (LSTM)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    fig2.tight_layout()

    return fig1, fig2, msg

def load_prophet(df, ticker):
    # Load model
    model_path = f"models/Prophet/{ticker.lower().replace('-', '_')}_prophet_model.pkl"
    model = joblib.load(model_path)

    # Prepare data
    df = df.reset_index()
    df['ds'] = pd.to_datetime(df['Date'])
    df['y'] = df[ticker]
    df = df[['ds', 'y']]

    # Split
    total_len = len(df)
    train_end = int(total_len * 0.70)
    val_end = int(total_len * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Predict
    forecast_val = model.predict(val_df[['ds']])
    forecast_test = model.predict(test_df[['ds']])

    # Evaluation function
    def evaluate(y_true, y_pred, label):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        msg = (f"{ticker} {label} (Prophet) - \n\n"
               f"RMSE (USD): {rmse:.4f},\n\n"
               f"MAE (USD): {mae:.4f},\n\n"
               f"MAPE (%): {mape:.2f}%")
        return msg

    # Evaluate
    msg1 = evaluate(val_df['y'].values, forecast_val['yhat'].values, "Validation")
    msg2 = evaluate(test_df['y'].values, forecast_test['yhat'].values, "Test")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_df['ds'], test_df['y'], label='Actual', color='black')
    ax.plot(test_df['ds'], forecast_test['yhat'], label='Predicted', color='blue')
    ax.set_title(f"{ticker} Prophet Forecast vs Actual (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, msg1, msg2

def load_xgb(df, ticker, use_diff=False):
    model = XGBRegressor()
    model.load_model(f"models/XGBoost/{ticker.lower().replace('-', '_')}_xgb_model.bin")

    # Differencing data as done in training (if use_diff=True)
    series = df[ticker].diff() if use_diff else df[ticker].copy()

    # Create lag features exactly like in training
    def create_lag_features(data, lags=5):
        df_lag = pd.DataFrame({'y': data})
        for lag in lags:
            df_lag[f'lag_{lag}'] = df_lag['y'].shift(lag)
        df_lag.dropna(inplace=True)  # Ensure no missing values
        return df_lag

    lags = [1, 2, 3, 4, 6]  # Same lags as in training
    data_lagged = create_lag_features(series, lags)

    n = len(data_lagged)

    # Ensure the same splitting logic as in training
    train_end = int(n * 0.7)  # 70% for training
    test_start = int(n * 0.85)  # 85% for test (same as in training code)

    # Ensure that the test set is the same (from 85% to 100%)
    test = data_lagged[test_start:]
    X_test, y_test = test.drop('y', axis=1), test['y']

    # Make predictions
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Reconstruct actual and predicted prices based on differenced data (if use_diff=True)
    if use_diff:
        # Correct start price based on test start + lag
        start_price = df[ticker].iloc[train_end + len(lags)]  # Correct start price based on lag
        predicted_prices = start_price + np.cumsum(y_pred)  # Cumulative sum for predicted prices
        actual_prices = start_price + np.cumsum(y_test.values)  # Actual prices
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    else:
        predicted_prices = y_pred
        actual_prices = y_test.values
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    msg = f"{ticker} Evaluation Metrics (XGBoost):\n\n - RMSE (USD): {rmse:.4f}\n\n - MAE (USD):  {mae:.4f}\n\n - MAPE (%): {mape:.2f}%"

    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_prices, label=f'Actual {ticker}')
    ax.plot(predicted_prices, label=f'Predicted {ticker}')
    ax.set_title(f"{ticker}: Actual vs Predicted (XGBoost)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, msg

def load_arima(df, ticker):
    currency_series = df[ticker].dropna()

    # Split data: 70% train, 15% test, 15% validation
    total_len = len(currency_series)
    train_end = int(total_len * 0.70)
    test_end = int(total_len * 0.85)

    train = currency_series[:train_end]
    test = currency_series[train_end:test_end]
    val = currency_series[test_end:]

    actual = pd.concat([test, val])

    model = joblib.load(f"models/ARIMA/{ticker.lower().replace('-', '_')}_arima_model.pkl")

    n_periods = len(test) + len(val)
    forecast = model.predict(n_periods=n_periods)

    forecast_index = currency_series.index[train_end:]
    forecast_series = pd.Series(forecast, index=forecast_index)

    mse = mean_squared_error(actual, forecast_series)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, forecast_series)
    mape = np.mean(np.abs((actual - forecast_series) / actual.replace(0, np.nan))) * 100

    msg = f"{ticker} Evaluation Metrics (ARIMA):\n\nRMSE: {rmse:.2f}\n\nMAE: {mae:.2f}\n\nMAPE: {mape:.2f}%"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual.index, actual, label="Actual")
    ax.plot(forecast_series.index, forecast_series, label="Predicted", linestyle="--")
    ax.set_title(f"{ticker}: Actual vs Predicted (ARIMA)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, msg

def forecast_future_price(model, scaler, data, seq_len=60, future_day=7):
    if isinstance(model, Prophet):
        return forecast_prophet(model, data, future_day)
    elif isinstance(model, xgb.Booster):
        return forecast_xgboost(model, data, lags=[1, 2, 3, 4, 6], future_day=future_day)
    elif isinstance(model, pm.arima.arima.ARIMA) or hasattr(model, 'predict_in_sample'):
        return forecast_arima(model, data, future_day)
    elif hasattr(model, 'predict'):
        return forecast_lstm(model, scaler, data, seq_len, future_day)
    else:
        raise ValueError("Unsupported model type. Must be one of: LSTM, Prophet, XGBoost, ARIMA.")

def forecast_lstm(model, scaler, data, seq_len=60, future_day=7):
    if len(data) < seq_len:
        raise ValueError(f"Not enough data to create sequence of length {seq_len}.")

    data_scaled = scaler.transform(data.reshape(-1, 1))
    current_seq = data_scaled[-seq_len:].reshape(1, seq_len, 1)

    predictions = []

    for _ in range(future_day):
        pred = model.predict(current_seq, verbose=0)
        predictions.append(pred[0][0])
        pred_reshaped = pred.reshape(1, 1, 1)
        current_seq = np.append(current_seq[:, 1:, :], pred_reshaped, axis=1)

    predictions = np.array(predictions).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predictions).flatten()

    future_dates = [datetime.today() + timedelta(days=i + 1) for i in range(future_day)]
    return build_result(predicted_prices, future_dates)

def forecast_prophet(model, data, future_day=7):
    future_dates = pd.date_range(datetime.today(), periods=future_day + 1).tolist()
    future_df = pd.DataFrame({'ds': future_dates[1:]})

    df = pd.DataFrame({
        'ds': pd.date_range(datetime.today() - timedelta(days=len(data)), periods=len(data), freq='D'),
        'y': data
    })

    forecast = model.predict(future_df)
    predicted_prices = forecast['yhat'].values
    return build_result(predicted_prices, future_dates[1:])

def forecast_xgboost(model, data, lags=[1, 2, 3, 4, 6], future_day=7):
    max_lag = max(lags)
    if len(data) < max_lag:
        raise ValueError(f"Not enough data to apply lag features: need at least {max_lag} points.")

    data_extended = list(data[-max_lag:].copy())
    predictions = []

    for _ in range(future_day):
        features = [data_extended[-lag] for lag in lags]
        dmatrix = xgb.DMatrix(np.array([features]))
        pred = model.predict(dmatrix)[0]
        predictions.append(pred)
        data_extended.append(pred)  # Extend with predicted value for recursive lags

    predicted_prices = np.array(predictions)
    future_dates = [datetime.today() + timedelta(days=i + 1) for i in range(future_day)]

    return build_result(predicted_prices, future_dates)

def forecast_arima(model, data, future_day=7):
    forecast = model.predict(n_periods=future_day)
    future_dates = [datetime.today() + timedelta(days=i + 1) for i in range(future_day)]
    return build_result(forecast, future_dates)

def build_result(prices, dates):
    min_price = prices.min()
    max_price = prices.max()
    min_date = dates[prices.argmin()]
    max_date = dates[prices.argmax()]
    return {
        "prices": prices,
        "dates": dates,
        "min_price": min_price,
        "min_date": min_date,
        "max_price": max_price,
        "max_date": max_date
    }

def buy_sell_signal(df, ticker, future_day=7, model_type='LSTM'):
    recent_data = df[ticker].dropna().values[-100:]

    if model_type == 'LSTM':
        model, history, scaler = load_lstm_model(
            f"models/LSTM/{ticker.lower().replace('-', '_')}_lstm_model.keras",
            f"models/LSTM/{ticker}_lstm_history.pkl",
            f"models/LSTM/{ticker}_lstm_scaler.pkl"
        )
    elif model_type == 'Prophet':
        model = load_prophet_model(f"models/Prophet/{ticker.lower().replace('-', '_')}_prophet_model.pkl")
        history, scaler = None, None
    elif model_type == 'XGBoost':
        model = load_xgboost_model(f"models/XGBoost/{ticker.lower().replace('-', '_')}_xgb_model.bin")
        history, scaler = None, None
    elif model_type == 'ARIMA':
        model = load_arima_model(f"models/ARIMA/{ticker.lower().replace('-', '_')}_arima_model.pkl")
        history, scaler = None, None
    else:
        raise ValueError("model_type must be one of: 'LSTM', 'Prophet', 'XGBoost', 'ARIMA'")

    result = forecast_future_price(model, scaler, recent_data, seq_len=60, future_day=future_day)

    msg = f"{model_type} Forecast from now to {future_day} days ahead for {ticker}:\n\n - Lowest price (Buy): ${result['min_price']:.2f} on {result['min_date'].date()}\n\n - Highest price (Sell): ${result['max_price']:.2f} on {result['max_date'].date()}"

    return msg

def load_lstm_model(model_path, history_path, scaler_path):
    model = load_model(model_path)
    history = joblib.load(history_path)
    scaler = joblib.load(scaler_path)
    return model, history, scaler

def load_prophet_model(model_path):
    return joblib.load(model_path)

def load_xgboost_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def load_arima_model(model_path):
    return joblib.load(model_path)


# === Sidebar Navigation ===
st.sidebar.title("COM724 AE2 Software")
section = st.sidebar.radio("Navigate to", [
    "1. Overview",
    "2. K-Means Clustering",
    "3. Pearson correlation heatmap",
    "4. EDA with up-to-date dataset",
    "5. Model evaluation",
    "6. Buy and sell signals"
])

# === 1. Overview ===
if section.startswith("1"):
    df = data_collection(up_to_date=True)

    st.title("1. Dataset Overview")
    st.markdown(f"**Dataset shape and size:** {df.shape[0]} rows × {df.shape[1]} columns\n\n"
                f"Rows representing timestamps, and columns representing the closing prices of each cryptocurrencies.", unsafe_allow_html=True)

    st.subheader("Independent variables vs dependent variable (target)")
    st.markdown(f"**Independent variables** - timestamp\n\n"
                f"**Dependent variable (target)** - cryptocurrency closing prices")

    st.subheader("Dataset sample")
    st.dataframe(df.head(30))

    st.markdown(f"The dataset was downloaded using the yahoo finance (*yfinance*) python library.\n\n"
                f"The system passes its internal datetime to extract the exact data from one-year ago until today.\n\n"
                f"An API requests were also implemented to extract up-to-date top 30 cryptocurrencies from "
                f"*CoinGecko* API.\n\nFuthermore, currencies containing NaN values were replaced with those with no null values.", unsafe_allow_html=True)

# === 2. Bar plots/histogram ===
elif section.startswith("2"):
    st.title("2. K-Means Clustering")

    # Up-to-date clusters
    df = data_collection(up_to_date=True)
    st.subheader("Visualization of up-to-date cryptocurrency clusters")
    fig, cluster_groups = k_means_clustering(df)
    st.pyplot(fig)

    st.markdown(f"**Cluster group 1:** {cluster_groups[0]}\n\n"
                f"**Cluster group 2:** {cluster_groups[1]}\n\n"
                f"**Cluster group 3:** {cluster_groups[2]}\n\n"
                f"**Cluster group 4:** {cluster_groups[3]}\n\n<hr>", unsafe_allow_html=True)

    st.markdown(f"1. Min-Max scaling is commonly recommended before applying K-Means clustering, especially prior to dimensionality reduction, because it ensures all features are on the same scale. "
                f"StandardScaler is unnecessary in this case since the dataset contains only non-negative values, and preserving the original range is preferable.\n\n"
                f"2. Among dimensionality reduction techniques, Autoencoders (AE) are preferred over PCA, and PCA is preferred over LDA. "
                f"This is because the dataset exhibits non-linear relationships, which PCA cannot capture since it only handles linear transformations. "
                f"Autoencoders are better suited for such non-linear patterns and perform well in unsupervised settings. LDA, on the other hand, requires class labels or a target variable, which are not available in this unsupervised context.\n\n"
                f"3. The encoding dimension in the Autoencoder was set to 4, reducing the original 365 features to 4. "
                f"Reducing it to fewer than the number of clusters (e.g., 2) results in an error because K-Means requires enough dimensions to define distinct clusters.\n\n"
                f"4. K-Means clustering was chosen because the project specified four cluster groups in advance. "
                f"This aligns well with K-Means, which requires the number of clusters to be defined before training.\n\n<hr>", unsafe_allow_html=True)

    # Clusters at the time of the project
    st.markdown("<h3>Visualization of cryptocurrency clusters at the time of the project<br>(from 08.05.2024 to 07.05.2025)</h3>", unsafe_allow_html=True)
    st.image('plots/cluster.png')

    cluster_groups = {
        0: [
            'AAVE-USD', 'ADA-USD', 'AVAX-USD', 'BGB-USD', 'CRO-USD', 'DAI-USD',
            'DOGE-USD', 'DOT-USD', 'ETC-USD', 'GT-USD', 'HBAR-USD', 'ICP-USD',
            'LEO-USD', 'LINK-USD', 'LTC-USD', 'NEAR-USD', 'OKB-USD',
            'ONDO-USD', 'SHIB-USD', 'SOL-USD', 'TON-USD', 'TRX-USD',
            'USDC-USD', 'USDT-USD', 'WBT-USD'
        ],
        3: ['BCH-USD', 'BNB-USD'],
        1: ['BTC-USD'],
        2: ['ETH-USD', 'STETH-USD']
    }

    st.markdown(f"**Cluster group 1:** {cluster_groups[0]}\n\n"
                f"**Cluster group 2:** {cluster_groups[1]}\n\n"
                f"**Cluster group 3:** {cluster_groups[2]}\n\n"
                f"**Cluster group 4:** {cluster_groups[3]}\n\n<hr>", unsafe_allow_html=True)

    st.markdown(f"The chosen cryptocurrencies for the project were: CRO-USD, BTC-USD, STETH-USD, and BCH-USD")

# === 3. Pearson correlation heatmap ===
elif section.startswith("3"):
    st.title("3. Pearson correlation heatmap")
    st.markdown("<h3>Correlation heatmap with up-to-date dataset<br>and selected cryptocurrency</h3>", unsafe_allow_html=True)

    df = data_collection(up_to_date=True)
    ticker_list = df.columns

    # Assuming ticker_list is already defined
    selected_ticker = st.selectbox("Select a cryptocurrency ticker:", ticker_list)
    fig = top_correlated(df, selected_ticker, 4, True, 'inferno')
    st.pyplot(fig)

    st.markdown("<h3>Correlation heatmap of selected cryptocurrencies at the time of the project<br>(from 08.05.2024 to 07.05.2025)</h3>", unsafe_allow_html=True)
    st.image('plots/heatmap_BCH-USD.png')
    st.image('plots/heatmap_BTC-USD.png')
    st.image('plots/heatmap_CRO-USD.png')
    st.image('plots/heatmap_STETH-USD.png')

# === 4. EDA with up-to-date dataset ===
elif section.startswith("4"):
    # --- Load Data ---
    st.title("4. EDA with Up-to-Date Dataset")
    df = data_collection(up_to_date=True)

    # --- EDA Options ---
    eda_options = [
        "Time-Series Decomposition",
        "Rolling Mean and Standard Deviation",
        "Volatility Heatmap",
        "Relative Performance (Base-100)"
    ]
    selected_eda = st.selectbox("Select an EDA technique:", eda_options)

    # --- Decomposition ---
    if selected_eda == "Time-Series Decomposition":
        st.markdown("### Time-Series Decomposition")
        ticker = st.selectbox("Select ticker for decomposition", df.columns)
        period = st.slider("Seasonal period (e.g. 7 for weekly)", min_value=2, max_value=30, value=7)
        result = seasonal_decompose(df[ticker], model='additive', period=period)
        fig = result.plot()
        for ax in fig.axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        st.pyplot(fig)

    # --- Rolling Mean and Std ---
    elif selected_eda == "Rolling Mean and Standard Deviation":
        st.markdown("### Rolling Mean and Standard Deviation")
        window = st.slider("Rolling Window (Days)", min_value=2, max_value=30, value=7)
        ticker_rm = st.selectbox("Select ticker for rolling analysis", df.columns)
        rolling_mean = df[ticker_rm].rolling(window).mean()
        rolling_std = df[ticker_rm].rolling(window).std()
        fig, ax = plt.subplots()
        ax.plot(df[ticker_rm], label='Original')
        ax.plot(rolling_mean, label='Rolling Mean')
        ax.plot(rolling_std, label='Rolling Std')
        ax.set_title(f"{ticker_rm} - Rolling Mean and Std")
        ax.legend()
        st.pyplot(fig)

    # --- Volatility Heatmap ---
    elif selected_eda == "Volatility Heatmap":
        st.markdown("### Volatility Heatmap")
        window_vol = st.slider("Volatility Rolling Window", min_value=2, max_value=30, value=7)
        volatility = df.rolling(window=window_vol).std()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(volatility.T, cmap='coolwarm', cbar_kws={'label': 'Volatility'}, ax=ax)
        ax.set_title("Rolling Volatility of Tickers")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.xticks(rotation=0)
        st.pyplot(fig)

    # --- Base-100 Normalized Chart ---
    elif selected_eda == "Relative Performance (Base-100)":
        st.markdown("### Relative Performance (Base-100 Indexing)")
        indexed = df / df.iloc[0] * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        indexed.plot(ax=ax, legend=False)
        ax.set_title("Normalized Price Performance (Base-100)")
        ax.set_ylabel("Normalized Price")
        st.pyplot(fig)

# === 5. Model evaluation ===
elif section.startswith("5"):
    st.title("5. Model evaluation")
    st.markdown("<h3>Evaluation of the models trained for the<br>buy and sell signals function</h3>",
                unsafe_allow_html=True)

    df = data_collection(up_to_date=False)

    # Define the lists for models and tickers
    models_list = ['LSTM', 'Prophet', 'XGBoost', 'ARIMA']
    tickers_list = ['BCH-USD', 'BTC-USD', 'CRO-USD', 'STETH-USD']

    # Create two columns
    col1, col2 = st.columns(2)

    # Place the selectboxes in the respective columns
    with col1:
        selected_model = st.selectbox("Select a model:", models_list)

    with col2:
        selected_ticker = st.selectbox("Select a ticker:", tickers_list)

    if selected_model == 'LSTM':
        fig1, fig2, msg = load_lstm(df, selected_ticker, seq_len=60)
        st.pyplot(fig1)
        st.markdown("<hr>>", unsafe_allow_html=True)
        st.pyplot(fig2)
        st.markdown(f"<hr>{msg}", unsafe_allow_html=True)
    elif selected_model == 'Prophet':
        fig, msg1, msg2 = load_prophet(df, selected_ticker)
        st.pyplot(fig)
        st.markdown(f"<hr>{msg1}", unsafe_allow_html=True)
        st.markdown(f"<hr>{msg2}", unsafe_allow_html=True)
    elif selected_model == 'XGBoost':
        use_diff = False
        if selected_ticker == 'STETH-USD':
            use_diff=True
        fig, msg = load_xgb(df, selected_ticker, use_diff=use_diff)
        st.pyplot(fig)
        st.markdown(f"<hr>{msg}", unsafe_allow_html=True)
    elif selected_model == 'ARIMA':
        fig, msg = load_arima(df, selected_ticker)
        st.pyplot(fig)
        st.markdown(f"<hr>{msg}", unsafe_allow_html=True)

# === 6. Buy and sell signals ===
elif section.startswith("6"):
    st.title("6. Buy and sell signals")
    st.markdown("<h3>Generate buy and sell signals using trained models</h3>",
                unsafe_allow_html=True)

    df = data_collection(up_to_date=False)

    # Define the lists for models and tickers
    models_list = ['LSTM', 'Prophet', 'XGBoost', 'ARIMA']
    tickers_list = ['BCH-USD', 'BTC-USD', 'CRO-USD', 'STETH-USD']

    # Create two columns
    col1, col2 = st.columns(2)

    # Place the selectboxes in the respective columns
    with col1:
        selected_model = st.selectbox("Select a model:", models_list)

    with col2:
        selected_ticker = st.selectbox("Select a ticker:", tickers_list)

    st.subheader("Select number of days (in the future):")

    number_of_days = 1

    col3, col4 = st.columns([2, 1])
    with col3:
        # Textbox for number of days (number_input ensures only numbers)
        num_days = st.number_input("Days", min_value=1, step=1, label_visibility="collapsed")
    with col4:
        # Button to trigger action
        if st.button("Submit"):
            number_of_days = num_days

    msg = buy_sell_signal(df, selected_ticker, number_of_days, model_type=selected_model)

    st.markdown(f"<hr>{msg}", unsafe_allow_html=True)