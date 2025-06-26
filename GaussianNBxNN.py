import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
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
LOOKBACK = 7
LOOKAHEAD = 1
expected_return = 2
LEARNING_RATE = 0.1
EPOCHS = 100
DROPOUT = 0.3
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
    data["overn"] = np.where(data["return"] > expected_return, 1, 0)
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
#data = VWAP(data)
data = compute_atr(data)
#data = compute_rsi(data)
data = return_over_period(data)
data = add_pct_chng(data)
data = over_npct_span(data)
data.dropna(inplace=True)

feature_columns = data.columns.difference(["overn", "return", "high", "low"])
x = data[feature_columns].to_numpy()
y = data["overn"].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False)

scaler = StandardScaler()

x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class GaussianNBLayer(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_classes = num_classes
    self.num_features = num_features

    self.means = nn.Parameter(torch.randn(num_classes, num_features))
    self.log_vars = nn.Parameter(torch.zeros(num_classes, num_features))
    self.log_priors = nn.Parameter(torch.zeros(num_classes))

  def forward(self, x):
    means = self.means.unsqueeze(0)
    vars = torch.exp(self.log_vars).unsqueeze(0)
    x = x.unsqueeze(1)

    log_likelyhood = -0.5 * torch.sum(
        torch.log(2 * 3.14159265359 * vars) + ((x - means) ** 2) / vars,
        dim=2
    )

    log_posteriors = self.log_priors + log_likelyhood
    return log_posteriors


class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, x):
    return self.layers(x)

class HybrinMLPGaussianNB(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_features, num_classes):
    super().__init__()
    self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    self.nb = GaussianNBLayer(num_classes=num_classes, num_features=num_features)

  def forward(self, x):
    log_probs = self.nb(x)
    return self.mlp(log_probs)

#tensor b shape -> num_features
num_features = x_train.shape[1]
num_classes = 2
input_dim= num_classes
output_dim = num_classes
hidden_dim = 256

model = HybrinMLPGaussianNB(input_dim, hidden_dim, output_dim, num_features, num_classes)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()




for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(x_train)
    loss = loss_fn(logits, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            model.eval()
            train_preds = torch.argmax(model(x_train), dim=1)
            train_acc = (train_preds == y_train_tensor).float().mean().item()

            test_preds = torch.argmax(model(x_test), dim=1)
            test_acc = (test_preds == y_test_tensor).float().mean().item()

            print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


