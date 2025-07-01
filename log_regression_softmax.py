import autobnn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, r2_score
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# ------------ config --------------
SYMBOL = "BTC-USD"
START = "2015-01-01"
END = "2025-04-17"
PERIOD = "1d"
BATCH_SIZE = 32
LOOKBACK = 7
expected_return = 2
LEARNING_RATE = 0.001
EPOCHS = 1000
DROPOUT = 0.3
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
data['target'] = data['future_return'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
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

mlr = multinomial_lr()
mlr.train()
mlr.eval()


