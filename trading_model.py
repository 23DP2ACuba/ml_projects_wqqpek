from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pandas_ta as ta
import yfinance as yf
import pandas as pd

# ---------- config ----------
lookahead = 2
target_pct = 2
lookback = 2
symbol = "BTC-USD"
period = "1d"
start="2020-01-01"
end="2025-06-01"
# ----------------------------
data = yf.Ticker(symbol).history(period=period, start=start, end=end)
def get_features(data):
  data["HLpct"] = ((data.High - data.Low) / data.High) * 100
  data["PCT_Change"] = ((data.Close - data.Open) / data.Open) * 100

  data["SMA_100"] = ta.ema(data.Close, length=100)
  data["SMA_200"] = ta.ema(data.Close, length=200)

  data["ADX"] = ta.adx(high=data["High"], low=data["Low"], close=data["Close"])["ADX_14"]

  data["ATR"] = ta.atr(high=data["High"], low=data["Low"], close=data["Close"])

  data["VWAP"] = ta.vwap(high=data["High"], low=data["Low"], close=data["Close"], volume=data["Volume"])
  global features
  features = ["HLpct", "PCT_Change", "SMA_100", "SMA_200", "ADX", "ATR", "VWAP"]

  for feature in features:
      data[f"{feature}_rolling_{lookback}"] = data[feature].rolling(window=lookback).mean()
  return data

def get_target(data):
  target = pd.DataFrame(data["Close"])

  for i in range(1, lookahead+1):
    target[f"Close{i}"] = target["Close"].shift(-i)

  pct_changes = [(target[f"Close{i}"] / target["Close"] - 1) * 100 for i in range(1, lookahead + 1)]
  data["Target"] = pd.concat(pct_changes, axis=1).ge(target_pct).any(axis=1).astype(int)
  data.dropna(inplace=True)
  return data
data = get_features(data)
data = get_target(data)
x = data[data.columns.difference(["Target", "Close"])]
y = data["Target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)

cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KMeans(n_clusters=3))
])

cluster_pipeline.fit(x)

x_train["market_state"] = cluster_pipeline.predict(x_train)
x_test["market_state"] = cluster_pipeline.predict(x_test)

svc_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(max_iter=10000))
])

svc_pipeline.fit(x_train, y_train)
y_pred = svc_pipeline.predict(x_test)
print("Accuracy score: ", accuracy_score(y_test, y_pred))

#Test
l = []
for i in range(200, 1, -1):
  test_data = x_test.tail(i)
  test_target = y_test.tail(i)
  test_data = test_data.head(1)
  test_target = test_target.head(1)
  y_pred = pipeline.predict(test_data)
  l.append(y_pred.item() == test_target.item())
  print(f"pred: {y_pred}, y_act: {test_target.item()}, {y_pred.item() == test_target.item()}")

import collections
collections.Counter(l)
