from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.functional as F
import yfinance as yf
from torch import nn
import numpy as np
import torch
import ta

ASSET = "EURUSD=X"
START = "2020-01-01"
END = "2025-09-01"
SEQ_LEN = 30
PRED_LEN = 1
BATCH_SIZE = 32
LR = 3e-4


def load_data(start, end, asset):
  df = yf.Ticker(asset).history(interval="1d", start = start, end=end)
  return df

def create_features(df):

  df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
  bbands = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
  df["bb_high"], df["bb_low"] = bbands.bollinger_hband(), bbands.bollinger_lband()

  df["ma"] = df["Close"].rolling(window=20).mean()
  df["ma_slope"] = df["ma"].diff()

  df.fillna(method="bfill", inplace=True)
  df.fillna(method="ffill", inplace=True)

  return df

def preprocessing(df, cols=None, test_size=0.2):
  if not cols:
    cols = list(df.columns)

  target_idx = cols.index("Close")

  data = df[cols].values
  x_train = data[:int(len(data)*(1-test_size))]
  x_test = data[int(len(data)*(1-test_size)+1):]
  print(x_test)
  scaler = MinMaxScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)

  return x_train_scaled, x_test_scaled, target_idx

class DataStore(Dataset):
  def __init__(self, data, seq_len=60, pred_len=1, feature_dim=4, target_col_idx=3):
    self.data = data
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.feature_dim = feature_dim
    self.target_col_idx = target_col_idx

  def __len__(self):
    return len(self.data) - self.seq_len - self.pred_len+1

  def __getitem__(self, idx):
    x = self.data[idx: idx + self.seq_len]

    y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_col_idx]

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    if self.pred_len == 1:
        y = y.squeeze()

    return x, y



class TsFT(nn.Module):
  def __init__(self, feature_size=9, num_layers=2, d_model=64, nhead=8, dim_ffd=256, dropout=0.1, seq_len=60, pred_len=1):
    super(TsFT, self).__init__()
    self.fc_in = nn.Linear(feature_size, d_model)

    self.pos_embd = nn.Parameter(torch.zeros(1, seq_len, d_model))

    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ffd,
        dropout=dropout,
        activation="gelu"
    )

    self.transformer_encoder = nn.TransformerEncoder(
        enc_layer, num_layers=num_layers
    )

    self.fc_out = nn.Linear(d_model, pred_len)


  def forward(self, x):
    batch_size, seq_len, _ = x.shape
    x = self.fc_in(x)
    x += self.pos_embd[:, :seq_len, :]
    x = x.permute(1, 0, 2)
    x = self.transformer_encoder(x)
    x = x[-1, :, :]          # take the last time step -> (batch, d_model)
    x = self.fc_out(x)       # (batch, pred_len)
    return x


def train(model, train_loader, val_loader=None, lr=3e-4, epochs=20, device="cpu"):
  criterion = nn.MSELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

  for epoch in range(epochs):
    model.train()
    total_loss=[]
    for xi, yi in train_loader:
      xi.to(device)
      yi.to(device)

      optimizer.zero_grad()

      out = model(xi)
      loss = criterion(out, yi)
      loss.backward()
      optimizer.step()
      total_loss.append(loss.item())

    print(f"Epoch: {epoch+1}, Train Loss: {sum(total_loss)/len(total_loss)}")
    if val_loader:
      eval_loss=[]
      model.eval()
      with torch.no_grad():
        for xi, yi in val_loader:
          xi.to(device)
          yi.to(device)

          out = model(xi)
          loss = criterion(out, yi)
          train_losses.append(loss.item())
        print(f"Eval Loss: {sum(total_loss)/len(total_loss)}")
  return model

def eval_model(model, test_loader, scaler, cols, target_col_idx, window_size=10, start_idx=0, pred_len=1, device="cpu"):
  model.eval()
  act_vals = []
  pred_vals = []

  with torchno_grad():
    for xi, yi in val_loader:
        xi.to(device)
        yi.to(device)

        pred = model(xi).cpu().numpy()
        yi = yi.cpu().numpy()

        for i in range(len(pred)):
          tmp_pred = np.zeros((pred_len,len(cols)))
          tmp_pred[:, target_col_idx] = pred[i]

          tmp_act = np.zeros((pred_len,len(cols)))
          tmp_act[:, target_col_idx] = pred[i]

          tmp_pred = scaler.inverse_transform(tmp_pred)[:, target_col_idx]
          tmp_act = scaler.inverse_transform(tmp_act)[:, target_col_idx]

          pred_vals.extend(tmp_pred)
          act_vals.extend(tmp_act)

    act_vals = np.array(act_vals).flatten()
    pred_vals = np.array(pred_vals).flatten()

    mse = np.mean((act_vals - pred_vals)**2)
    mae = np.mean(abs(act_vals - pred_vals))

    print(f"MSE:{mse}")
    print(f"MAE:{mae}")

    if start_idx < 0 or start_idx > len(atc_vals):
      start_idx = 0

    end_idx = min(start_idx + window_size*pred_len, len(act_vals))

    plt.plot(range(start_idx, end_idx), act_vals[start_idx:end_idx])
    plt.plot(range(start_idx, end_idx), pred_vals[start_idx:end_idx])
    plt.legend("actual prices", "predicte prices")

    plt.show()


df = load_data(start=START, end=END, asset=ASSET)
df = create_features(df)

x_train, x_test, target_col_idx = preprocessing(df)



trainds = DataStore(x_train, SEQ_LEN, PRED_LEN, len(df.columns), target_col_idx)
testds= DataStore(x_test, SEQ_LEN, PRED_LEN, len(df.columns), target_col_idx)

train_loader = DataLoader(trainds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testds, batch_size=BATCH_SIZE, shuffle=False)

model = TsFT(
    feature_size=len(df.columns),
    num_layers=2,
    d_model=64,
    nhead=8,
    dim_ffd=256,
    dropout=0.1,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN
)

device = "cuda" if torch.cuda.is_available() else "cpu"
trained_model = train(model, train_loader, test_loader, lr=LR, epochs=20, device=device)

