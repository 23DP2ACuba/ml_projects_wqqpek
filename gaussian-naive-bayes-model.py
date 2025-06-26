import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------ config --------------
SYMBOL = "BTC-USD"
START = "2015-01-01"
END = "2025-04-17"
PERIOD = "1d"
BATCH_SIZE = 32
LOOKBACK = 5
LOOKAHEAD = 5
expected_return = 2
LEARNING_RATE = 0.001
EPOCHS = 100
DROPOUT = 0.2
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# ----------------------------------


data = yf.Ticker(SYMBOL).history(period=PERIOD, start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]
data.columns = data.columns.str.lower()

def add_lookback(data):
    for i in range(LOOKBACK):
        data[f"close{i+1}"] = data['close'].shift(i+1)
    return data

def add_pct_chng(data):
  data["HL_pct"] = (data.high - data.low) / data.close * 100
  return data

def return_over_period(data):
    data["return"] = (data["close"].shift(-LOOKAHEAD) - data["close"]) / data["close"] * 100
    return data

def over_npct_span(data):
    over_threshold = []
    for i in range(1, 1 + LOOKAHEAD):
        future_return = (data["close"].shift(-i) - data["close"]) / data["close"] * 100
        over_threshold.append(future_return > expected_return)
    
    data["overn"] = np.any(over_threshold, axis=0).astype(int)
    return data

def over_npct(data):
    data["overn"] = np.where(data["return"] > 2.0, 1, 0)
    return data


def VWAP(data):
    data["vwap"] = (((data['high'] + data['close'] + data['low']) / 3) * data['volume']).cumsum() / data['volume'].cumsum()
    return data

def compute_atr(data, period=14):
    high = data['high']
    low = data['low']
    close = data['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low), 
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    data['atr'] = atr
    return data



data = add_lookback(data)
data = return_over_period(data)
data = VWAP(data)
data = compute_atr(data)
#data = compute_rsi(data)
data = return_over_period(data)
data = add_pct_chng(data)
data = over_npct(data)
data.dropna(inplace=True)

feature_columns = data.columns.difference(["overn", "return", "high", "low"])
x = data[feature_columns].to_numpy()
y = data["overn"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GaussianNB())
])



pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
print(accuracy_score(y_pred, y_test))
