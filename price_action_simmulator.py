import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ------------ Config --------------
SYMBOL = "BTC-USD"
START = "2015-01-01"
END = "2025-04-17"
PERIOD = "1d"
SIMLENGTH = 1
LOOKBACK = 5
# ----------------------------------

# Download historical data
data = yf.Ticker(SYMBOL).history(period=PERIOD, start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]
data.columns = data.columns.str.lower()

# ---------- Feature Creators ----------

def create_close_features(data):
    features = ["close"]
    for i in range(1, LOOKBACK + 1):
        lag_col = f"close-{i}"
        data[lag_col] = data["close"].shift(i)
        features.append(lag_col)

    data["target"] = data["close"].shift(-SIMLENGTH)
    data.dropna(inplace=True)
    return data, features


def create_hl_pct_features(data):
    features = ["close"]
    for i in range(1, LOOKBACK + 1):
        lag_col = f"close-{i}"
        data[lag_col] = data["close"].shift(i)
        features.append(lag_col)

    data["target"] = (data["high"] - data["low"]) / data["close"]
    data.dropna(inplace=True)
    return data, features

def create_vol_pct_features(data):
    features = ["close", "volume", "target"]
    data["target"] = data["volume"].shift(-SIMLENGTH)
    data.dropna(inplace=True)
    return data, features

# ---------- Training & Simulation Class ----------

class TrainMarketSim:
    def close_sim(self, data, features):
        x = data[features]
        y = data["target"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        self.close_model = TransformedTargetRegressor(
            regressor=LinearRegression(),
            transformer=StandardScaler()
        )
        self.close_model.fit(x_train, y_train)

        y_pred = self.close_model.predict(x_test)
        print("Close R2 Score:", r2_score(y_test, y_pred))

        self.close_features = features
        self.last_close_row = data.iloc[-1]
        return self.close_model

      


    def hl_pct_sim(self, data, features):
        x = data[features]
        y = data["target"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        self.hl_model = TransformedTargetRegressor(
            regressor=LinearRegression(),
            transformer=StandardScaler()
        )
        self.hl_model.fit(x_train, y_train)

        y_pred = self.hl_model.predict(x_test)
        print("Volume R2 Score:", r2_score(y_test, y_pred))

        self.hl_features = features
        self.last_hl_row = data.iloc[-1]
        return self.hl_model

    def generate_sequence(self, max_steps=20, noise_std=0.0):
      close_history = list(self.last_close_row[self.close_features].values.flatten())
      hl_history = list(self.last_hl_row[self.hl_features].values.flatten())

      simulated_data = []

      for step in range(max_steps):
          close_input = np.array(close_history).reshape(1, -1)
          hl_input = np.array(hl_history).reshape(1, -1)

          next_close = self.close_model.predict(close_input)[0]
          next_hl_pct = self.hl_model.predict(hl_input)[0]

          next_close += np.random.normal(loc=0.3, scale=noise_std)
          next_hl_pct += np.random.normal(loc=0.0, scale=noise_std / 100) 

          simulated_data.append({
              "step": step,
              "predicted_close": next_close,
              "predicted_hl_pct": next_hl_pct
          })

          close_history = [next_close] + close_history[:-1]
          hl_history = [next_close] + hl_history[:-1]

      return pd.DataFrame(simulated_data)

# ---------- Run Training and Simulation ----------
train = TrainMarketSim()

close_data, close_features = create_close_features(data.copy())
train.close_sim(close_data, close_features)

hl_data, hl_features = create_hl_pct_features(data.copy())
train.hl_pct_sim(hl_data, hl_features)



simulated_prices = train.generate_sequence(max_steps=20, noise_std=2)
print(simulated_prices)
