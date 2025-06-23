from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Sequential
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import numpy as np

ticker = "BTC-USD"
df = yf.Ticker(ticker).history(period="2y")
df["High_Low_Perc"] = ((df["High"] - df["Low"]) / df["Low"]) * 100

rsi_periods = [5, 10, 14, 21, 30]
for period in rsi_periods:
    df[f"RSI_{period}"] = ta.rsi(df["Close"], length=period)
df.dropna(inplace=True)

rsi_columns = [f"RSI_{p}" for p in rsi_periods]
X_rsi = df[rsi_columns].values

scaler_rsi = StandardScaler()
X_rsi_scaled = scaler_rsi.fit_transform(X_rsi)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_rsi_scaled)
df["PCA_1"] = X_pca[:, 0]
df["PCA_2"] = X_pca[:, 1]

df["Target"] = df["Close"].shift(-1) - df["Close"]

df.dropna(inplace=True)

features = ["Close", "Volume", "High_Low_Perc", "PCA_1", "PCA_2"]
df_features = df[features]
Target = df["Target"]

scaler_features = StandardScaler()
df_features_scaled = pd.DataFrame(scaler_features.fit_transform(df_features), columns=features, index=df_features.index)

backcandles = 10 
x, y = [], []
sample_dates = df_features_scaled.index[backcandles:]

for i in range(backcandles, len(df_features_scaled)):
    x.append(df_features_scaled.iloc[i-backcandles:i].values)
    y.append(Target.iloc[i])
  
x, y = np.array(x), np.array(y)

split_index = int(0.8 * len(x))
X_train, X_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
test_dates = sample_dates[split_index:]

model = Sequential([
    layers.Input(shape=(backcandles, len(features))),
    layers.LSTM(150, activation="linear"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
history = model.fit(X_train, y_train, batch_size=32, epochs=10, shuffle=False, verbose=2)

predictions = model.predict(X_test)

test_open_prices = df.loc[test_dates, "Close"].values
actual_pct_returns = y_test / test_open_prices

signals_ls = np.where(predictions.flatten() > 0, 1, -1)
signals_long_only = np.where(predictions.flatten() > 0, 1, 0)

strategy_returns_ls = signals_ls * actual_pct_returns * 2
strategy_returns_long_only = signals_long_only * actual_pct_returns * 2

buy_hold_returns = actual_pct_returns

cumulative_ls = np.cumprod(1 + strategy_returns_ls) - 1
cumulative_long_only = np.cumprod(1 + strategy_returns_long_only) - 1
cumulative_buy_hold = np.cumprod(1 + buy_hold_returns) - 1

final_roi_ls = cumulative_ls[-1] * 100
final_roi_long_only = cumulative_long_only[-1] * 100
final_roi_buy_hold = cumulative_buy_hold[-1] * 100

print("ROI for x2 Leveraged Long-Short Strategy: {:.2f}%".format(final_roi_ls))
print("ROI for x2 Leveraged Long-Only Strategy: {:.2f}%".format(final_roi_long_only))
print("ROI for Buy & Hold Strategy: {:.2f}%".format(final_roi_buy_hold))


trade_amount = 1000
pnl_ls = trade_amount * strategy_returns_ls
pnl_long = trade_amount * strategy_returns_long_only
pnl_buy_hold = trade_amount * buy_hold_returns


total_pnl_ls = np.sum(pnl_ls)
total_pnl_long = np.sum(pnl_long)
total_pnl_buy_hold = np.sum(pnl_buy_hold)

avg_pnl_ls = np.mean(pnl_ls)
avg_pnl_long = np.mean(pnl_long)
avg_pnl_buy_hold = np.mean(pnl_buy_hold)

print("Total Profit/Loss for x2 Leveraged Long-Short Strategy: ${:.2f}".format(total_pnl_ls))
print("Average Profit/Loss per trade for x2 Leveraged Long-Short Strategy: ${:.2f}".format(avg_pnl_ls))
print("Total Profit/Loss for x2 Leveraged Long-Only Strategy: ${:.2f}".format(total_pnl_long))
print("Average Profit/Loss per trade for x2 Leveraged Long-Only Strategy: ${:.2f}".format(avg_pnl_long))
print("Total Profit/Loss for Buy & Hold Strategy: ${:.2f}".format(total_pnl_buy_hold))
print("Average Profit/Loss per trade for Buy & Hold Strategy: ${:.2f}".format(avg_pnl_buy_hold))

plt.figure(figsize=(12, 6))
plt.plot(test_dates, cumulative_ls, label="x2 Leveraged Long-Short Strategy")
plt.plot(test_dates, cumulative_long_only, label="x2 Leveraged Long-Only Strategy")
plt.plot(test_dates, cumulative_buy_hold, label="Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Backtesting Performance")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
