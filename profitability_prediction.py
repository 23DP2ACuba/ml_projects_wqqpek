from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import yfinance as yf
import pandas as pd
import numpy as np

# ---------- config ----------
lookahead = 2
target_pct = 2
lookback = 2
symbol = "BTC-USD"
period = "1d"
start = "2020-01-01"
end = "2025-06-01"
WINDOW_SIZE = 20
TRADING_FEES = 0.002
LOOKBACK = 8
MA_PERIOD = 20
N_DAYS = 5
THRESHOLD = 0.03
RSI_WINDOW = 14
# ----------------------------

data = yf.Ticker(symbol).history(interval=period, start=start, end=end)

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
    df["Target"]=0
    df.loc[df['Future_Return'] > THRESHOLD, "Target"] = 1
    #df.loc[df['Future_Return'] < -THRESHOLD, "Target"] = -1

    return df.dropna()

data = create_features(data)

x = data[data.columns.difference(["Target", "Close", "Future_Return", "Stock Splits"])]
y = data["Target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True)

cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KMeans(n_clusters=5))
])

cluster_pipeline.fit(x_train)
x_train = x_train.copy()
x_test = x_test.copy()
x_train["market_state"] = cluster_pipeline.predict(x_train)
x_test["market_state"] = cluster_pipeline.predict(x_test)

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.92)), 
    ("model", RandomForestClassifier(n_estimators=1500))
])

rf_pipeline.fit(x_train, y_train)
y_pred = rf_pipeline.predict(x_test)
print("Accuracy score:", accuracy_score(y_test, y_pred))

x_test["TradeTF"] = y_pred
x_train["TradeTF"] = rf_pipeline.predict(x_train)
x_train["TradeTF"].value_counts()

def simulate_trades(df, entry_col="TradeTF", sl_pct=0.02, tp_pct=0.05, n_days=10):
    df = df.copy()

    new_target = []

    for i in range(len(df) - n_days):
        if df.iloc[i][entry_col] != 1:
            new_target.append(np.nan)
            continue

        entry_price = df.iloc[i + 1]["Open"]  
        high_window = df["High"].iloc[i + 1:i + 1 + n_days].values
        low_window = df["Low"].iloc[i + 1:i + 1 + n_days].values

        hit_tp = np.any((high_window - entry_price) / entry_price >= tp_pct)
        hit_sl = np.any((entry_price - low_window) / entry_price >= sl_pct)

        if hit_tp and not hit_sl:
            new_target.append(1)
        elif hit_sl and not hit_tp:
            new_target.append(0)
        elif hit_tp and hit_sl:
            tp_index = np.argmax((high_window - entry_price) / entry_price >= tp_pct)
            sl_index = np.argmax((entry_price - low_window) / entry_price >= sl_pct)
            new_target.append(1 if tp_index <= sl_index else 0)
        else:
            new_target.append(np.nan)

    df["Target"] = new_target + [np.nan] * n_days
    return df.dropna()


