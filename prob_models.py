import autobnn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, r2_score
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

# ------------ config --------------
SYMBOL = "BTC-USD"
START = "2015-01-01"
END = "2025-04-17"
PERIOD = "1d"
BATCH_SIZE = 32
LOOKBACK = 7
LEARNING_RATE = 0.001
EPOCHS = 1000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# ----------------------------------

features = [
    'log_return_5',
    'momentum_3',
    'volatility',
    'volume',
    'rolling_return_5'
]

data = yf.Ticker(SYMBOL).history(period=PERIOD, start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]
data.columns = data.columns.str.lower()

for i in range(1, LOOKBACK+1):
  idx = f"close-{i}"
  data[idx] = data.close.shift(i)
  features.append(idx)
data["rolling_return_5"] = data["close"].pct_change(3)
data["log_return_5"] = np.log(data['close'] / data['close'].shift(5))
data['log_return'] = np.log(data['close'] / data['close'].shift(1))
data["volatility"] = data["log_return"].rolling(window=5).std()
data["momentum_3"] = data['close'] - data['close'].shift(5)

data['future_return'] = data['close'].shift(-5) / data['close'] - 1
threshold = 0.002
data['target'] = data['future_return'].apply(lambda x: 1 if x > threshold else (2 if x < -threshold else 0))
data.dropna(inplace=True)

x = data[features]
y = data.target
print(y.value_counts(normalize=True))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

class multinomial_lr():
  def __init__(self):
    self.pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
    ])
  def train(self):
    self.pipeline.fit(x_train, y_train)
  def eval(self):
    y_pred = self.pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"model r2_score: {accuracy}")
  def pred(self, x):
    probs = self.pipeline.predict_proba(x)

class MCDropoutNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop = nn.Dropout(p=0.1 or 0.0)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
      x = self.relu(self.drop(self.fc1(x)))
      return self.fc2(x)
    
    def process(self, x, y):
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
      scaler = StandardScaler()
      x_train = scaler.fit_transform(x_train)
      x_test = scaler.transform(x_test)

      x_train = torch.tensor(x_train, dtype=torch.float32)
      x_test = torch.tensor(x_test, dtype=torch.float32)
      y_train = torch.tensor(y_train.values, dtype=torch.long)
      y_test = torch.tensor(y_test.values, dtype=torch.long)

      return x_train, x_test, y_train, y_test

    def load(self, x_train, x_test, y_train, y_test, BATCH_SIZE):
      train_dataset = TensorDataset(x_train, y_train)
      test_dataset = TensorDataset(x_test, y_test)
      train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
      test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
      return train_loader, test_loader
