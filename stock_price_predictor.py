"""
Stock pricepredictor
"""
#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import yfinance as yf


df = yf.Ticker("ETH-USD").history(period = "1d", start = "2021-06-01", end = "2025-05-15")
df = df[["Open", "High",	"Low",	"Close", 	"Volume"]]


df["Return"] = (df.Close - df.Open) / df.Open
df["Volatility"] = df.Return.rolling(window=14).std()
df["HL_pct"] = ((df.High - df.Low) / df.High) * 100
df["TP"] = ((df.High + df.Low + df.Close) / 3) * df.Volume
df["VWAP"] = df["TP"].rolling(window=14).sum() / df["Volume"].rolling(window=14).sum()
df = df[["VWAP", "HL_pct", "Volatility", "Return", "Close", "Volume"]]

df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

x = df[df.columns.difference(["Target"])]
y = df["Target"]
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20)
#x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=.25)

pipeline = Pipeline([
    ('pca', PCA(n_components=5)),
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(
        n_estimators=10000,
        max_depth=5,
        learning_rate=0.01,
        verbosity = 1
    ))
])

pipeline.fit(x_train, y_train)

#%%
df = yf.Ticker("ETH-USD").history(period = "1d", start = "2025-04-1", end = "2025-06-15")
df = df[["Open", "High",	"Low",	"Close", 	"Volume"]]

end = "2025-05-12"
target = "2025-05-7"

df["Return"] = (df.Close - df.Open) / df.Open
df["Volatility"] = df.Return.rolling(window=14).std()
df["HL_pct"] = ((df.High - df.Low) / df.High) * 100
df["TP"] = ((df.High + df.Low + df.Close) / 3) * df.Volume
df["VWAP"] = df["TP"].rolling(window=14).sum() / df["Volume"].rolling(window=14).sum()
df = df[['Close', 'HL_pct', 'Return', 'VWAP', 'Volatility', 'Volume']]


row = df.loc[[end]]
y_pred = pipeline.predict(row)
print(*y_pred)
