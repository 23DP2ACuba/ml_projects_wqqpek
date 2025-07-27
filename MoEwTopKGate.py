"""
Mixture of Experts (MoE) Architecture
"""

import math
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- Config ----------------
device = torch.device('cpu')
symbol = "BTC-USD"
start = "2020-01-01"
end = "2025-06-01"
period = "1d"
WINDOW_SIZE = 20
TRADING_FEES = 0.002
LOOKBACK = 8
MA_PERIOD = 20
N_DAYS = 5
THRESHOLD = 0.03
NUM_CLASSES = 2
NUM_MLPS = 4
HIDDEN_DIM = 128
DROPOUT = 0.4
BATCH_SIZE = 32
epochs = 100
LR = 1e-3
RSI_WINDOW = 14
LABEL_SMOOTHING = 0.1

# ---------------- Data ----------------
def create_features(data):
    df = data.copy()
    df['Return'] = df['Close'].pct_change().shift(1)
    df['Volatility'] = df['Return'].rolling(5, min_periods=1).std().shift(1)
    df['Momentum'] = df['Close'].shift(1) - df['Close'].shift(6)
    df['Log_Volume'] = np.log(df['Volume'].shift(1))
    df["Ma"] = df['Close'].rolling(MA_PERIOD).mean()
    df['Cl_to_Ma_pct'] = (df['Close'] - df['Ma']) / df['Close'] * 100
    df["Z-Score"] = (df['Return'] - df['Return'].rolling(WINDOW_SIZE).mean()) / df['Return'].rolling(WINDOW_SIZE).std()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=RSI_WINDOW).mean()
    loss = -delta.clip(upper=0).rolling(window=RSI_WINDOW).mean()
    rs = gain / (loss + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))
    for i in range(0, LOOKBACK + 1, 5):
        df[f"Ma_t-{i}"] = df["Ma"].shift(i)
    df['Future_Return'] = (df['Close'].shift(-N_DAYS) - df['Close']) / df['Close']
    df["Target"] = 0
    df.loc[df['Future_Return'] > THRESHOLD, "Target"] = 1
    return df.dropna()

raw_data = yf.Ticker(symbol).history(interval=period, start=start, end=end)
data = create_features(raw_data)
features = data.drop(columns=["Target", "Future_Return"])
targets = data["Target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features.values)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, targets.values, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                               torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
                               torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ---------------- Model ----------------
def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

class Experts(nn.Module):
    def __init__(self, dim, num_experts=16, hidden_dim=None, activation=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else dim * 4
        w1 = init_(torch.zeros(num_experts, dim, hidden_dim))
        w2 = init_(torch.zeros(num_experts, hidden_dim, dim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('ebd,edh->ebh', x, self.w1)
        hidden = self.act(hidden)
        return torch.einsum('ebh,ehd->ebd', hidden, self.w2)

class TopKGate(nn.Module):
    def __init__(self, model_dim, num_experts, k=1):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.w_gating = nn.Linear(model_dim, num_experts)

    def forward(self, x, mask=None):
        logits = self.w_gating(x)
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        topk_scores = F.softmax(topk_vals, dim=-1)
        return topk_idx, topk_scores

class MoE(nn.Module):
    def __init__(self, dim, hidden_dim=None, num_experts=16, activation=nn.GELU):
        super().__init__()
        self.num_experts = num_experts
        self.experts = Experts(dim, num_experts, hidden_dim, activation)
        self.gate = TopKGate(model_dim=dim, num_experts=num_experts)

    def forward(self, x):
        B, _, D = x.shape
        topk_idx, topk_scores = self.gate(x) 
        topk_idx = topk_idx.squeeze(1) 
        topk_scores = topk_scores.squeeze(1)

        expert_inputs = x.repeat(self.num_experts, 1, 1).transpose(0, 1) 
        expert_inputs = expert_inputs.reshape(self.num_experts, -1, D) 

        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.permute(1, 0, 2) 

        selected = []
        for i in range(topk_idx.shape[1]):
            indices = topk_idx[:, i].unsqueeze(1).unsqueeze(2).expand(-1, 1, D)
            selected.append(torch.gather(expert_outputs, 1, indices).squeeze(1) * topk_scores[:, i:i+1])

        out = torch.stack(selected, dim=0).sum(dim=0)
        return out

# ---------------- Training ----------------
model = MoE(dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, num_experts=NUM_MLPS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def test():
  with torch.no_grad():
    loss_sum = []
    for idx, (data, targets) in enumerate(test_loader):
          data = data.to(device)
          targets = targets.to(device)
          logits = model(data)
          loss = criterion(logits, targets)
          loss_sum.append(loss.item())
    print(f" Loss: {min(loss_sum):.4f}")

model.train()
for epoch in range(epochs):
    total_loss = 0
    print(f"Epoch {epoch+1} >>>", end=" ")
    for idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        logits = model(data)

        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if idx % 10 == 0:
            test()
            print("|", end="")

    print(f" Loss: {total_loss:.4f}")

