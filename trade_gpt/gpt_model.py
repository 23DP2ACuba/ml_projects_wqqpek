"""
GPT-model.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
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
LOOKBACK = 7
LOOKAHEAD = 3
expected_return = 2
LEARNING_RATE = 0.001
n_embd = 384
block_size = 8
max_iters = 1000
eval_iter = 100
DROPOUT = 0.3
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# ----------------------------------


data = yf.Ticker(SYMBOL).history(period=PERIOD, start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]


def create_features(data):
    lookback = 6
    if len(data) < lookback + 1:
        raise ValueError(f"Need at least {lookback+1} periods for features")

    data["Return"] = data["Close"].pct_change().shift(1).mean()
    data["Volatility"] = data["Return"].rolling(5, min_periods=1).std()
    data["Momentum"] = data["Close"].shift(1) - data["Close"].shift(6)
    data["Log_Volume"] = np.log(data["Volume"].shift(1) + 1e-6)

    def adjustable_rolling_mean(series, max_window=None):
      if max_window is None or max_window == "max":
          return series.expanding(min_periods=1).mean()
      else:
          return series.rolling(window=max_window, min_periods=1).mean()

    data["STA"] = adjustable_rolling_mean(data["Volume"], max_window=5)
    data["LTA"] = adjustable_rolling_mean(data["Volume"], max_window=10)
    data["VO"] = (data["STA"] - data["LTA"]) / data["LTA"]

    return data.dropna()

data = create_features(data)
features = [
    "Open", "High", "Low", "Close", "Volume",         
    "Return", "Volatility", "Momentum", "Log_Volume", "VO" 
]

data = data[features]

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KMeans(n_clusters=5))
])

cluster_pipeline.fit(train_data)

train_data["market_state"] = cluster_pipeline.predict(train_data)
val_data["market_state"] = cluster_pipeline.predict(val_data)

features = train_data.columns

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.fit_transform(val_data)

train_data = torch.tensor(train_data, dtype=torch.float32)
val_data = torch.tensor(val_data, dtype=torch.float32)

# ---- features -------
n_input_features = len(features)
# ---------------------

def get_batch(train=True):
    data_split = train_data if train else val_data
    ix = torch.randint(len(data_split) - block_size, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

get_batch()
