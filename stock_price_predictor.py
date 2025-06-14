from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import yfinance as yf


df = yf.Ticker("ETH-USD").history(period = "1d", start = "2021-06-01", end = "2025-06-01")
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
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=.20)

pipeline = Pipeline([
    ('pca', PCA(n_components=5)),
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(
        n_estimators=10000,
        max_depth=5,
        learning_rate=0.01
    ))
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
accuracy = r2_score(y_test, y_pred)
print(f"model r2_score: {accuracy}")
