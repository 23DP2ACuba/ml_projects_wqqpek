
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from sklearn.pipeline import Pipeline
from datetime import datetime
import pandas_ta as ta
import yfinance as yf
import numpy as np

START = "2010-01-01"
END = datetime.now().strftime("%Y-%m-%d")
SYMBOL = "SPY"
LOOKBACK = 10
MA_LENS = [50, 100, 200]
THRESHOLD = 0.02
N_DAYS = 5
THRESHOLD = 0.05
SL = 0.02  

class DecisionTreeModel:
  def __init__(self, ma_lens=[10], lookback=10, use_scaler=True, threshold=0.5, simple_target=True, sl = None, start = "2010-01-01",symbol="SPY",  end=datetime.now().strftime("%Y-%m-%d")):
    df = yf.Ticker(SYMBOL).history(start=START, end=END, interval="1d")
    df = df.drop(["Dividends", "Stock Splits", "Capital Gains"], axis=1, errors="ignore")

    vix = yf.Ticker("^VIX").history(start=START, end=END, interval="1d")
    
    df["VIX_Close"] = vix["Close"].values
    self.data = df

    self.ma_lens = ma_lens
    self.pipeline = Pipeline([
      ("model", DecisionTreeClassifier(random_state=42))
    ])
    if use_scaler:
      self.pipeline = Pipeline([
      ("scaler", StandardScaler()),
      #("pca", PCA(n_components=0.92)),
      ("model", DecisionTreeClassifier())
    ])
    else:
        self.pipeline = Pipeline([
      ("model", DecisionTreeClassifier(random_state=42))
    ])

    self.lookback = lookback
    self.scaler = StandardScaler()
    self.features_list = []
    self.targets_list = []

    assert(type(simple_target)==type(True))
    self.simple_target = simple_target
    self.threshold = threshold
    self.sl = sl

  def label_takeprofit_stoploss(self, prices):
    n = len(prices)
    targets = np.zeros(n, dtype=int)

    for i in range(n):
        base_price = prices[i]
        take_profit_price = base_price * (1 + self.threshold)
        stop_loss_price = base_price * (1 - self.sl)

        future_prices = prices[i+1:]

        hit_sl = np.where(future_prices <= stop_loss_price)[0]
        hit_tp = np.where(future_prices >= take_profit_price)[0]

        if hit_tp.size == 0 and hit_sl.size == 0:
            continue

        if hit_tp.size == 0:
            continue

        if hit_sl.size == 0:
            targets[i] = 1
            continue

        if hit_tp[0] < hit_sl[0]:
            targets[i] = 1  
        else:
            pass

    return targets

  def get_features(self):
    df = self.data.copy()

    for i in range(1, self.lookback+1):
      df[f"Close-{i}"] = df["Close"].shift(i)

    for period in self.ma_lens:
      df[f"SMA_{period}"]  = ta.sma(self.data["Close"], length=period)

    vol_average = df["Volume"].rolling(window=20).mean().std()
    vol_change = (df["Volume"].shift(1) - df["Volume"]) / df["Volume"]
    df["VolChangeToAverage"] = vol_change / vol_average

    df["ATR"] = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=14)

    df["MACD_LINE"], df["Hist"], df["SIG"]  = ta.macd(df['Close']).values.T
    df['MACD'] = 0
    df.loc[df['MACD_LINE'] > df["SIG"], 'MACD'] = 1
    df.drop(["MACD_LINE", "SIG"], axis=1, inplace = True)
    
    df['RSI'] = ta.rsi(df['Close'], length=14)
    print(df)
    if self.simple_target:
      df['Future_Return'] = (df['Close'].shift(-N_DAYS) - df['Close']) / df['Close']
      df["Target"]=0
      df.loc[df['Future_Return'] > THRESHOLD, "Target"] = 1
      

    else:
      assert(self.sl is not None)
      df['Target'] = self.label_takeprofit_stoploss(df['Close'].values)
    return df.dropna()

  def train_model(self, test_size=0.2):
    self.data = self.get_features()
    x = self.data[self.data.columns.difference(["Target"])]
    y = self.data["Target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

    self.pipeline.fit(x_train, y_train)

    features = x.columns
    

    y_pred = self.pipeline.predict(x_test)
    print("Accuracy score:", accuracy_score(y_test, y_pred))

    print("Classification Report:\n", classification_report(y_test, y_pred))

    return {"clf": self.pipeline, "features": features, "data": self.data}

    
    
  def predict(self):
    pass
    



model = DecisionTreeModel(
    lookback=LOOKBACK,
    simple_target=False, 
    sl=SL, 
    threshold=THRESHOLD, 
    ma_lens=MA_LENS, 
    symbol=SYMBOL, 
    start=START, 
    end=END
)

output_data = model.train_model()
