# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:37:20 2025

@author: SRIKANTH
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from statsmodels.tsa.seasonal import seasonal_decompose
import dtale
import sweetviz as sv
from sklearn.preprocessing import MinMaxScaler
#loading the csv file
data=pd.read_csv(r"D:\project\Data_set_Iron.csv")


#Auto EDA for primary analysis
#using dtale
d=dtale.show(data)
d.open_browser()

#using sweetviz
s=sv.analyze(data)
s.show_html()

#data dictionary
'''
1.Date-The date of the recorded iron price. Format: MM/DD/YYYY.(object)

2.Price-The closing price of iron on the given date.(float)

3.Open-The opening price of iron on the given date.(float)

4.High-The highest recorded price of iron on the given date.(float)

5.Low-The lowest recorded price of iron on the given date.(float)

6.Vol.-Trading volume (mostly missing values, possible issues with formatting).(object)

7.Change %-Percentage change in the price from the previous day (stored as a string with %, needs conversion).(object)

'''
#mysql credentials to access db
user = 'root'
password = 'Password'
db = 'Iron_ore_db'
#connection string to connect to mysql
engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{db}")
#loading the dataset to mysql
data.to_sql('Iron_ore_price3', con = engine, if_exists = 'replace',chunksize = 1000, index = False )

sql = 'select * from Iron_ore_price3'
df= pd.read_sql_query(text(sql), con = engine.connect())
df.head()

##############################################################
#Data Cleaning

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# Convert 'Change %' from string to float
df['Change %'] = df['Change %'].str.replace("%", "").astype(float)

# Drop 'Vol.' column due to excessive missing values
df.drop(columns=["Vol."], inplace=True)

# Forward fill missing Date values
df['Date'].fillna(method='ffill', inplace=True)

################################################################
#data transforamtion

df = df.sort_values("Date")
df.set_index("Date", inplace=True)
df.drop_duplicates(inplace=True)


features = ['Open', 'High', 'Low', 'Change %']
target = 'Price'

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = df.copy()
df_scaled[features + [target]] = scaler.fit_transform(df[features + [target]])

################################################################
#EDA


print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


#First Moment Business Decision
for x in df.columns:
    print(df[x].mean(),df[x].median(),df[x].mode())
  #  print(df[x].median())
  #  print(df[x].mode())

#Second Moment Business Decision
#variance
for x in df.columns:
    print(x,df[x].var())
#standard Deviation
for x in df.columns:
    print(x,df[x].std())
#Range   
for x in df.columns:
    print(x,max(df[x])-min(df[x]))


#Third Moment Business Decision
# Skewness check
#-0.5 to 0.5 → Approximately symmetric
#-1 to -0.5 or 0.5 to 1 → Moderately skewed
#Less than -1 or greater than 1 → Highly skewed
skewness_values = df.skew()
print("\nSkewness of each numeric column:")
print(skewness_values)


#Fourth Moment Business Decision

for x in df.columns:
    print(x,df[x].kurt())

# Histograms
df.hist(figsize=(10,6), bins=30, edgecolor="black")
plt.suptitle("Histograms of Numeric Columns", fontsize=14)
plt.show()

# Boxplots to check for outliers
df.plot(kind='box', subplots=True, sharey=False, figsize=(18,10))
plt.suptitle("Outliers Detection", fontsize=14)
plt.subplots_adjust(wspace=0.75)
plt.show()


# Line plot of Iron Ore Price over time
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Price'], label='Iron Ore Price', color='blue', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Iron Ore Price Over Time')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for relationships between numerical features
sns.pairplot(df, diag_kind="kde")
plt.suptitle("Pair Plot Analysis", fontsize=14)
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

decomposition = seasonal_decompose(df['Price'], model='additive', period=30)  # Adjust period based on data frequency


# Check for Stationarity (ADF Test)
# p-value < 0.05: Data is stationary.
# p-value > 0.05: Data has a trend or seasonality (needs differencing or transformations).

result = adfuller(df['Price'])
print("ADF Test Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("The data is stationary (no trend/seasonality).")
else:
    print("The data is non-stationary (trend/seasonality present).")

#Detect Seasonality using Autocorrelation (ACF Plot)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.figure(figsize=(10, 5))
plot_acf(df['Price'], lags=30)
plt.title("Autocorrelation Plot (Seasonality Check)")
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(df['Price'], lags=30)
plt.title("Autocorrelation Plot (Seasonality Check)")
plt.show()

# Calculate rolling mean for smoothing
df['rolling_mean'] = df['Price'].rolling(window=30).mean()

plt.figure(figsize=(10,6))
plt.plot(df.index, df['Price'], label='Iron Ore Price')
plt.plot(df.index, df['rolling_mean'], label='30-Day Rolling Mean', color='red')
plt.title('30-Day Rolling Mean of Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(df["Price"], label="Original Data")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label="Trend", color="red")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label="Seasonality", color="green")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label="Residual (Noise)", color="gray")
plt.legend()

plt.tight_layout()
plt.show()

############################## Model Building #################################

##################################  Deep Learning Models ###########################


####################### LSTM MODEL ########################################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import numpy as np


# Function to create LSTM sequences
def create_sequences(data, target_col, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length, target_col])
    return np.array(sequences), np.array(targets)

seq_length = 30
data_values = df_scaled[features + [target]].values
X, y = create_sequences(data_values, target_col=len(features), seq_length=seq_length)

# Train-Test Split
train_size = int(len(X) * 0.6)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
def build_lstm_model(units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Train Model
model = build_lstm_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# Make Predictions
y_pred = model.predict(X_test)

# Inverse Transform Predictions
y_test_actual = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1,1)), axis=1))[:, -1]
y_pred_actual = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]

# Evaluation Metrics
r2 = r2_score(y_test_actual, y_pred_actual)
mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)

print(f"R² Score: {r2}")
print(f"MAPE: {mape}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


# Compute Confidence Interval (95%)
confidence_level = 1.96  # For 95% confidence interval
y_pred_std = np.std(y_pred_actual - y_test_actual)
lower_bound = y_pred_actual - confidence_level * y_pred_std
upper_bound = y_pred_actual + confidence_level * y_pred_std

print("y_pred_std",y_pred_std)
print("lower_bound =",lower_bound)
print("upper_bound =",upper_bound)


# Plot Actual vs. Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-len(y_test_actual):], y_test_actual, label="Actual Price", color='blue')
plt.plot(df['Date'][-len(y_pred_actual):], y_pred_actual, label="Predicted Price", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Iron Ore Price")
plt.title("LSTM Model - Actual vs. Predicted Price")
plt.legend()
plt.show()

import joblib

# Save the MinMaxScaler for future inverse transformations
joblib.dump(scaler, "scale_lstm.pkl")

print("Model and scaler saved successfully.")

# While saving the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
model.save("lstm_model.h5")





import matplotlib.pyplot as plt

# Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label="Training Loss", color="blue")
plt.plot(history.history['val_loss'], label="Validation Loss", color="red", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.show()


'''
🔹 Hyperparameters Optimized
LSTM Units: Number of units in each LSTM layer.
Dropout Rate: Prevents overfitting by randomly dropping neurons.
Learning Rate: Controls how quickly the model updates weights.
Batch Size: Number of samples per gradient update.

 Why Use Hyperparameter Tuning?
1.Better Accuracy: Finds optimal LSTM settings.
2.Prevents Overfitting: Tuned dropout rates improve generalization.
3.Speeds Up Training: Avoids inefficient manual tuning.

'''


#4.

####################  XGBoost Hyperparameter Tuning #########################

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import joblib
# Load Dataset
df = pd.read_csv(r"Data_set_Iron.csv")

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)
df = df.drop(columns=['Vol.'])
df['Change %'] = df['Change %'].str.replace('%', '').astype(float)

features = ['Open', 'High', 'Low', 'Change %']
target = 'Price'

# Feature Engineering - Creating Lag Features
df['Price_Lag1'] = df['Price'].shift(1)   # 1-day lag
df['Price_Lag2'] = df['Price'].shift(2)   # 2-day lag
df['Rolling_Mean_3'] = df['Price'].rolling(window=3).mean()  # 3-day moving average
df.dropna(inplace=True)  # Remove NaN values from shifting

# Scaling Features (Only Features, Not Target)
scaler = MinMaxScaler(feature_range=(0, 1))
df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']] = scaler.fit_transform(df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']])

# Train-Test Split
X = df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']]
y = df[target].values  # Keep as raw values for inverse transformation later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# Save the trained model using pickle
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Save the MinMaxScaler for future use
with open("scaler_xgboost.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")

# Load the model (for future use)
with open("xgb_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Make Predictions
y_pred = loaded_model.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R² Score: {r2}")
print(f"MAPE: {mape}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Plot Actual vs. Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-len(y_test):], y_test, label="Actual Price", color='blue')
plt.plot(df['Date'][-len(y_pred):], y_pred, label="Predicted Price", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Iron Ore Price")
plt.title("XGBoost Model - Actual vs. Predicted Price")
plt.legend()
plt.show()



#5.Transformer
########################### TRANSFORMER MODEL #################################
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error


# Define Features & Target
features = ['Open', 'High', 'Low', 'Change %']
lag_features = ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']
all_features = features + lag_features
target = 'Price'

# Feature Engineering
df['Price_Lag1'] = df['Price'].shift(1)
df['Price_Lag2'] = df['Price'].shift(2)
df['Rolling_Mean_3'] = df['Price'].rolling(window=3).mean()

# Handle missing values from lagging
df[lag_features] = df[lag_features].fillna(method='bfill')

# Drop NaN rows after shifting
df.dropna(inplace=True)

# Scaling Features
scaler = MinMaxScaler(feature_range=(0, 1))
df[all_features] = scaler.fit_transform(df[all_features])
joblib.dump(scaler, "trans_scaler.pkl")

# Train-Test Split
X = df[all_features]
y = df[target].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Prepare Sequential Data for Transformer
seq_length = 10
X_train_seq = np.array([X_train.values[i - seq_length:i] for i in range(seq_length, len(X_train))])
X_test_seq = np.array([X_test.values[i - seq_length:i] for i in range(seq_length, len(X_test))])
y_train_seq = y_train[seq_length:]
y_test_seq = y_test[seq_length:]

# Transformer Model
def build_transformer(input_shape, num_heads=4, ff_dim=64, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(inputs, inputs)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

model = build_transformer(input_shape=(seq_length, X_train.shape[1]))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
model.summary()

# Train Model
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=50, batch_size=32,
    validation_data=(X_test_seq, y_test_seq), verbose=1
)

# Save Model
model.save("transformer_model.h5")

# Make Predictions
y_pred_seq = model.predict(X_test_seq)

# Evaluation Metrics
r2 = r2_score(y_test_seq, y_pred_seq)
mape = mean_absolute_percentage_error(y_test_seq, y_pred_seq)
mse = mean_squared_error(y_test_seq, y_pred_seq)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-len(y_test_seq):], y_test_seq, label="Actual Price", color='blue')
plt.plot(df['Date'][-len(y_pred_seq):], y_pred_seq, label="Predicted Price", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Iron Ore Price")
plt.title("Transformer Model - Actual vs. Predicted Price")
plt.legend()
plt.grid()
plt.show()


#6.
################## GRU(Gated Recurrent Unit) ############################################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error



features = ['Open', 'High', 'Low', 'Change %']
target = 'Price'

# Feature Engineering - Creating Lag Features
df['Price_Lag1'] = df['Price'].shift(1)
df['Price_Lag2'] = df['Price'].shift(2)
df['Rolling_Mean_3'] = df['Price'].rolling(window=3).mean()
df.dropna(inplace=True)

df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']] = scaler.fit_transform(df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']])

# Save Scaler as Pickle
with open("gru_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train-Test Split
X = df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']]
y = df[target].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape for GRU (Need 3D shape: [samples, time-steps, features])
seq_length = 20  # Increased sequence length for better trend recognition
X_train_seq = np.array([X_train.values[i - seq_length:i] for i in range(seq_length, len(X_train))])
X_test_seq = np.array([X_test.values[i - seq_length:i] for i in range(seq_length, len(X_test))])
y_train_seq = y_train[seq_length:]
y_test_seq = y_test[seq_length:]

# Optimized GRU Model
model = Sequential([
    GRU(128, activation='tanh', return_sequences=True, input_shape=(seq_length, X_train.shape[1])),
    Dropout(0.3),
    GRU(64, activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(1)  # Output layer
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")

# Train Model with Hyperparameters
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100, batch_size=16,  # Reduced batch size for better gradient updates
    validation_data=(X_test_seq, y_test_seq),
    verbose=1
)

# Save Model as Pickle
joblib.dump(model, "gru_model.pkl")

# Make Predictions
y_pred_seq = model.predict(X_test_seq)

# Evaluation Metrics
r2 = r2_score(y_test_seq, y_pred_seq)
mape = mean_absolute_percentage_error(y_test_seq, y_pred_seq)
mse = mean_squared_error(y_test_seq, y_pred_seq)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot Actual vs. Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-len(y_test_seq):], y_test_seq, label="Actual Price", color='blue')
plt.plot(df['Date'][-len(y_pred_seq):], y_pred_seq, label="Predicted Price", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Iron Ore Price")
plt.title("Optimized GRU Model - Actual vs. Predicted Price")
plt.legend()
plt.grid()
plt.show()



################################################################################

#7.CNN-LSTM Hybrid Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error


# Feature Engineering - Creating Lag Features
df['Price_Lag1'] = df['Price'].shift(1)   # 1-day lag
df['Price_Lag2'] = df['Price'].shift(2)   # 2-day lag
df['Rolling_Mean_3'] = df['Price'].rolling(window=3).mean()  # 3-day moving average
df.dropna(inplace=True)  # Remove NaN values from shifting

# Scaling Features (Only Features, Not Target)
df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']] = scaler.fit_transform(df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']])

# Train-Test Split
X = df[features + ['Price_Lag1', 'Price_Lag2', 'Rolling_Mean_3']]
y = df[target].values.reshape(-1, 1)  # Keep as raw values for inverse transformation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape for CNN-LSTM (Need 3D shape: [samples, time-steps, features])
seq_length = 30 # Using past 10 days for prediction
X_train_seq = np.array([X_train.values[i - seq_length:i] for i in range(seq_length, len(X_train))])
X_test_seq = np.array([X_test.values[i - seq_length:i] for i in range(seq_length, len(X_test))])
y_train_seq = y_train[seq_length:]
y_test_seq = y_test[seq_length:]

# CNN-LSTM Hybrid Model
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, X_train.shape[1])),
    MaxPooling1D(pool_size=2),
    LSTM(64, activation='tanh', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

# Train Model
history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test_seq), verbose=1)

# Make Predictions
y_pred_seq = model.predict(X_test_seq)

# Evaluation Metrics
r2 = r2_score(y_test_seq, y_pred_seq)
mape = mean_absolute_percentage_error(y_test_seq, y_pred_seq)
mse = mean_squared_error(y_test_seq, y_pred_seq)
rmse = np.sqrt(mse)

print(f"R² Score: {r2}")
print(f"MAPE: {mape}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Plot Actual vs. Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-len(y_test_seq):], y_test_seq, label="Actual Price", color='blue')
plt.plot(df['Date'][-len(y_pred_seq):], y_pred_seq, label="Predicted Price", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Iron Ore Price")
plt.title("CNN-LSTM Hybrid Model - Actual vs. Predicted Price")
plt.legend()
plt.show()

import joblib

# Save the trained CNN-LSTM model
model.save("cnn_lstm_model.h5")

# Save the MinMaxScaler for inverse transformation
joblib.dump(scaler, "scaler_cnn_lstm.pkl")

print("CNN-LSTM model and scaler saved successfully.")

###############################################################################
######################  ARIMA AND SARIMA MODEL ################################

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


# Set Date as Index
df.set_index('Date', inplace=True)

# Selecting Target Variable
target = 'Price'
train_size = int(len(df) * 0.8)  # 80% train, 20% test
train, test = df[target][:train_size], df[target][train_size:]

### ARIMA Model ###
arima_order = (5, 1, 5)  # You can tune these parameters (p, d, q)
arima_model = ARIMA(train, order=arima_order)
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(len(test))

# Evaluation Metrics for ARIMA
r2_arima = r2_score(test, arima_pred)
mape_arima = mean_absolute_percentage_error(test, arima_pred)
mse_arima = mean_squared_error(test, arima_pred)
rmse_arima = np.sqrt(mse_arima)

print(f"ARIMA R² Score: {r2_arima}")
print(f"ARIMA MAPE: {mape_arima}")
print(f"ARIMA MSE: {mse_arima}")
print(f"ARIMA RMSE: {rmse_arima}")

# Plot Actual vs. Predicted (ARIMA)
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size:], test, label="Actual Price", color='blue')
plt.plot(df.index[train_size:], arima_pred, label="Predicted Price (ARIMA)", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Iron Ore Price")
plt.title("ARIMA Model - Actual vs. Predicted Price")
plt.legend()
plt.show()

### SARIMA Model ###
sarima_order = (5, 1, 5)  # ARIMA (p,d,q)
seasonal_order = (1, 1, 1, 12)  # (P,D,Q,s) - Adjust s based on seasonality

sarima_model = SARIMAX(train, order=sarima_order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit()
sarima_pred = sarima_fit.forecast(len(test))

# Evaluation Metrics for SARIMA
r2_sarima = r2_score(test, sarima_pred)
mape_sarima = mean_absolute_percentage_error(test, sarima_pred)
mse_sarima = mean_squared_error(test, sarima_pred)
rmse_sarima = np.sqrt(mse_sarima)

print(f"SARIMA R² Score: {r2_sarima}")
print(f"SARIMA MAPE: {mape_sarima}")
print(f"SARIMA MSE: {mse_sarima}")
print(f"SARIMA RMSE: {rmse_sarima}")

# Plot Actual vs. Predicted (SARIMA)
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size:], test, label="Actual Price", color='blue')
plt.plot(df.index[train_size:], sarima_pred, label="Predicted Price (SARIMA)", color='green', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Iron Ore Price")
plt.title("SARIMA Model - Actual vs. Predicted Price")
plt.legend()
plt.show()

###############################################################################

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt


# Selecting Target Variable
target = 'Price'
train_size = int(len(df) * 0.8)  # 80% train, 20% test
train, test = df[target][:train_size], df[target][train_size:]

### 1️⃣ Simple Moving Average (SMA) ###
window_size = 10
sma_pred = test.copy()
sma_pred[:] = train.rolling(window=window_size).mean().iloc[-1]

# Evaluation Metrics for SMA
r2_sma = r2_score(test, sma_pred)
mape_sma = mean_absolute_percentage_error(test, sma_pred)
mse_sma = mean_squared_error(test, sma_pred)
rmse_sma = np.sqrt(mse_sma)

print(f"SMA R² Score: {r2_sma}")
print(f"SMA MAPE: {mape_sma}")
print(f"SMA MSE: {mse_sma}")
print(f"SMA RMSE: {rmse_sma}")

# Plot Actual vs. SMA Forecast
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Actual Price", color='blue')
plt.plot(test.index, sma_pred, label="SMA Predicted Price", color='red', linestyle="dashed")
plt.title("Simple Moving Average - Actual vs. Predicted")
plt.legend()
plt.show()


### 2️⃣ Exponential Moving Average (EMA) ###
ema_span = 10
ema_pred = train.ewm(span=ema_span, adjust=False).mean().iloc[-1]
ema_pred_series = pd.Series([ema_pred] * len(test), index=test.index)

# Evaluation Metrics for EMA
r2_ema = r2_score(test, ema_pred_series)
mape_ema = mean_absolute_percentage_error(test, ema_pred_series)
mse_ema = mean_squared_error(test, ema_pred_series)
rmse_ema = np.sqrt(mse_ema)

print(f"EMA R² Score: {r2_ema}")
print(f"EMA MAPE: {mape_ema}")
print(f"EMA MSE: {mse_ema}")
print(f"EMA RMSE: {rmse_ema}")

# Plot Actual vs. EMA Forecast
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Actual Price", color='blue')
plt.plot(test.index, ema_pred_series, label="EMA Predicted Price", color='green', linestyle="dashed")
plt.title("Exponential Moving Average - Actual vs. Predicted")
plt.legend()
plt.show()


### 3️⃣ Holt’s Linear Trend Model ###
holt_model = Holt(train).fit(smoothing_level=0.8, smoothing_slope=0.2)
holt_pred = holt_model.forecast(len(test))

# Evaluation Metrics for Holt's Linear Trend Model
r2_holt = r2_score(test, holt_pred)
mape_holt = mean_absolute_percentage_error(test, holt_pred)
mse_holt = mean_squared_error(test, holt_pred)
rmse_holt = np.sqrt(mse_holt)

print(f"Holt’s Linear Trend R² Score: {r2_holt}")
print(f"Holt’s Linear Trend MAPE: {mape_holt}")
print(f"Holt’s Linear Trend MSE: {mse_holt}")
print(f"Holt’s Linear Trend RMSE: {rmse_holt}")

# Plot Actual vs. Holt Forecast
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Actual Price", color='blue')
plt.plot(test.index, holt_pred, label="Holt's Linear Trend Forecast", color='orange', linestyle="dashed")
plt.title("Holt’s Linear Trend Model - Actual vs. Predicted")
plt.legend()
plt.show()


### 4️⃣ Holt-Winters (Triple Exponential Smoothing) ###
holt_winters_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
holt_winters_pred = holt_winters_model.forecast(len(test))

# Evaluation Metrics for Holt-Winters
r2_holt_winters = r2_score(test, holt_winters_pred)
mape_holt_winters = mean_absolute_percentage_error(test, holt_winters_pred)
mse_holt_winters = mean_squared_error(test, holt_winters_pred)
rmse_holt_winters = np.sqrt(mse_holt_winters)

print(f"Holt-Winters R² Score: {r2_holt_winters}")
print(f"Holt-Winters MAPE: {mape_holt_winters}")
print(f"Holt-Winters MSE: {mse_holt_winters}")
print(f"Holt-Winters RMSE: {rmse_holt_winters}")

# Plot Actual vs. Holt-Winters Forecast
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Actual Price", color='blue')
plt.plot(test.index, holt_winters_pred, label="Holt-Winters Forecast", color='purple', linestyle="dashed")
plt.title("Holt-Winters Model - Actual vs. Predicted")
plt.legend()
plt.show()
