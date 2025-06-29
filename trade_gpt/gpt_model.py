"""
GPT-model.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
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
eval_iter = 10
eval_interval = 50
n_layer = 6
n_head = 6
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
val_data = scaler.transform(val_data)

train_data = torch.tensor(train_data, dtype=torch.float32)
val_data = torch.tensor(val_data, dtype=torch.float32)

# ---- features -------
n_input_features = len(features)
# ---------------------

def get_batch(train=True):
    data_split = train_data if train else val_data
    max_start = len(data_split) - block_size - 1
    ix = torch.randint(max_start, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data_split[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y

def get_eval_batch(seq_start=0):
    x = val_data[seq_start:seq_start+block_size].unsqueeze(0).to(device)
    y = val_data[seq_start+1:seq_start+block_size+1].unsqueeze(0).to(device)
    return x, y

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x):
    b, t, c = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
    wei = f.softmax(wei, dim=-1)
    v = self.value(x)
    out = wei @ v
    return out

class MultiHead(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(DROPOUT)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(n_embd * 4, n_embd),
        nn.Dropout(DROPOUT)
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_head, n_embd):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHead(n_head, head_size)
    self.fwd = FeedForward(n_embd)
    self.l1 = nn.LayerNorm(n_embd)
    self.l2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.l1(x))
    x = x + self.fwd(self.l2(x))
    return x

class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_projection = nn.Linear(n_input_features, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.block = nn.Sequential(*[Block(n_head, n_embd) for _ in range(n_layer)])
    self.ln = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, n_input_features) 
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.2)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.2)

  def forward(self, idx, target=None):
    b, t, f = idx.shape  
    tok_emd = self.input_projection(idx)  
    pos_emd = self.position_embedding_table(torch.arange(t, device=idx.device)).unsqueeze(0)  
    x = tok_emd + pos_emd  
    x = self.block(x)
    x = self.ln(x)
    logits = self.lm_head(x)  

    if target is not None:
        loss = F.mse_loss(logits, target)  
    else:
        loss = None

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = self(idx_cond)
        next_step = logits[:, -1:, :]
        idx = torch.cat((idx, next_step), dim=1)
    return idx

model = GPT().to(device)
