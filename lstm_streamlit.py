import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model

# Set Seaborn Style
sns.set_style("darkgrid")
sns.set_palette("muted")
plt.rcParams.update({'figure.facecolor': 'white'})

# Load Model and Scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Iron Ore Price Forecasting", layout="wide")
st.title("📈 Iron Ore Price Forecasting using LSTM")

# Sidebar - Database Configuration
st.sidebar.header("🔗 Database Configuration")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
host = st.sidebar.text_input("Host", value="localhost")
dbname = st.sidebar.text_input("Database Name")
tablename_train_test = st.sidebar.text_input("Train/Test Table Name")
tablename_forecast = st.sidebar.text_input("Forecast Table Name")
connect_btn = st.sidebar.button("Connect to Database")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
    
    # Feature Selection
    features = ['Open', 'High', 'Low', 'Change %']
    target = 'Price'
    df_scaled = df.copy()
    df_scaled[features + [target]] = scaler.transform(df[features + [target]])
    
    # Train/Test Data
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Forecasting
    days = st.sidebar.slider("📅 Select forecast days", 1, 30, 7)
    seq_length = 30
    data_values = df_scaled[features + [target]].values
    X_input = data_values[-seq_length:].reshape(1, seq_length, len(features) + 1)
    predictions = []
    
    for _ in range(days):
        pred = model.predict(X_input)[0][0]
        new_row = np.append(X_input[:, 1:, :], [[np.append(data_values[-1, :-1], pred)]], axis=1)
        X_input = new_row.reshape(1, seq_length, len(features) + 1)
        predictions.append(pred)
    
    predictions_actual = scaler.inverse_transform(
        np.concatenate((np.tile(data_values[-1, :-1], (days, 1)), np.array(predictions).reshape(-1, 1)), axis=1)
    )[:, -1]
    
    forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=days+1, freq='D')[1:]
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": predictions_actual})
    
    # Confidence Interval Calculation
    forecast_std = np.std(predictions_actual) * 0.1  # 10% of standard deviation as a rough CI estimate
    forecast_df['Lower Bound'] = forecast_df['Predicted Price'] - forecast_std
    forecast_df['Upper Bound'] = forecast_df['Predicted Price'] + forecast_std
    
    # Combined Train & Test Table
    train_test_df = pd.concat([train_data, test_data], ignore_index=True)
    
    # Display Tables
    st.subheader("📊 Train & Test Data Table")
    st.dataframe(train_test_df)
    
    st.subheader("📊 Forecasted Prices Table")
    st.dataframe(forecast_df)
    

    # Train & Test Plot with Confidence Interval
    st.subheader("📈 Train & Test Data Trend")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=train_data['Date'], y=train_data['Price'], label="Train Data", color='blue')
    sns.lineplot(x=test_data['Date'], y=test_data['Price'], label="Test Data", color='green')
    plt.fill_between(test_data['Date'], test_data['Price'] - test_data['Price'].std() * 0.1,
                     test_data['Price'] + test_data['Price'].std() * 0.1, color='green', alpha=0.2)
    plt.xlabel("Date", fontsize=12, fontweight='bold')
    plt.ylabel("Iron Ore Price", fontsize=12, fontweight='bold')
    plt.title("Train & Test Data with Confidence Interval", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
    
    # Forecasted Prices Plot with Confidence Interval
    st.subheader("📉 Forecasted Price Trend")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=forecast_df['Date'], y=forecast_df['Predicted Price'], label="Forecasted Price", linestyle='dashed', color='crimson', linewidth=2)
    plt.fill_between(forecast_df['Date'], forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='crimson', alpha=0.2)
    plt.xlabel("Date", fontsize=12, fontweight='bold')
    plt.ylabel("Iron Ore Price", fontsize=12, fontweight='bold')
    plt.title("Forecasted Prices with Confidence Interval", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
    
    # Save to Database
    if connect_btn and username and password and host and dbname and tablename_train_test and tablename_forecast:
        engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{dbname}")
        train_test_df.to_sql(tablename_train_test, con=engine, if_exists='replace', index=False)
        forecast_df.to_sql(tablename_forecast, con=engine, if_exists='replace', index=False)
        st.success("✅ Data saved successfully to MySQL database!")
