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

data = yf.Ticker(SYMBOL).history(period=PERIOD, start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]
data.columns = data.columns.str.lower()



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
        print("R2 Score:", r2_score(y_test, y_pred))

        self.features = features
        return self.close_model

    def open_sim(self, data, features):
        x = data[features]
        y = data["target"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        self.open_model = TransformedTargetRegressor(
            regressor=LinearRegression(),
            transformer=StandardScaler()
        )
        self.open_model.fit(x_train, y_train)

        y_pred = self.open_model.predict(x_test)
        print("R2 Score:", r2_score(y_test, y_pred))

        self.features = features
        return self.open_model
      

    def generate_sequence(self, data, max_steps=20):
        """
        Generate a sequence of future closing prices using the last known row
        """

        start_row = 
        history = list(start_row[self.features].values.flatten())  
        simulated_data = []

        for step in range(max_steps):
            close_input = np.array(history).reshape(1, -1)
            next_price = self.close_model.predict(x_input)[0]
            next_open = self.open_model.predict(x_input)[0]

            simulated_data.append({
                "step": step,
                "predicted_close": next_price,
                "predicted_open": next_open
            })

            history = [next_price] + history[:-1]

        return pd.DataFrame(simulated_data)

def create_close_features(data):
  features = ["close"]
  for i in range(1, LOOKBACK + 1):
      lag_col = f"close-{i}"
      data[lag_col] = data["close"].shift(i)
      features.append(lag_col)

  data["target"] = data["close"].shift(-SIMLENGTH)

  data.dropna(inplace=True)
  return data


def create_open_features(data):
  features = ["open"]
  for i in range(1, LOOKBACK + 1):
      lag_col = f"open-{i}"
      data[lag_col] = data["open"].shift(i)
      features.append(lag_col)

  data["target"] = data["open"].shift(-SIMLENGTH)

  data.dropna(inplace=True)
  return data


train = TrainMarketSim()
close_data = create_close_features(data)
train.close_sim(close_data, features)
open_data = create_close_features(data)
train.open_sim(open_data, features)

last_row = data.iloc[-1]
simulated_prices = train.generate_sequence(last_row, max_steps=20)

print(simulated_prices)
