"""
Stock pricepredictor
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
import xgboost as xgm
import yfinance as yf


df = yf.Ticker("ETH-USD").history(period = "1d", start = "2021-06-01", end = "2025-06-01")
df = df[["Open", "High",	"Low",	"Close", 	"Volume"]]


df["Return"] = (df.Close - df.Open) / df.Open
df["Volatility"] = df.Return.rolling(window=14).std()
df["HL_pct"] = ((df.High - df.Low) / df.High) * 100
df["TP"] = ((df.High + df.Low + df.Close) / 3) * df.Volume
df["VWAP"] = df.TP.cumsum() / df.Volume.cumsum()
df[["VWAP", "HL_pct", "Volatility", "Return", "Close", "Volume"]]
df.tail(5)
