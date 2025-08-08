import pandas_ta as ta
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime

START = "2000-01-01"
END = datetime.now().strftime("%Y-%m-%d")
SYMBOL = "SPY"
LOOKBACK = 10
MA_LENS = [50, 100, 200]
THRESHOLD = 0.05
N_DAYS = 5
SL = 0.02
FEE = 0.002  
SLIPPAGE = 0.002 

class DecisionTreeModel:
    def __init__(self, ma_lens=[10], lookback=10, use_scaler=True, threshold=0.5, simple_target=True, sl=None, start="2018-01-01", symbol="TSLA", end=datetime.now().strftime("%Y-%m-%d")):
        df = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        df = df.drop(["Dividends", "Stock Splits", "Capital Gains"], axis=1, errors="ignore")
        vix = yf.Ticker("^VIX").history(start=start, end=end, interval="1d")
        df["VIX_Close"] = vix["Close"].reindex(df.index, method='ffill').values
        self.data = df
        self.ma_lens = ma_lens
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=42))
        ]) if use_scaler else Pipeline([
            ("model", DecisionTreeClassifier(random_state=42))
        ])
        self.lookback = lookback
        self.threshold = threshold
        self.sl = sl
        self.simple_target = simple_target

    def label_takeprofit_stoploss(self, prices):
        n = len(prices)
        targets = np.zeros(n, dtype=int)
        for i in range(n):
            base_price = prices[i]
            take_profit_price = base_price * (1 + self.threshold)
            stop_loss_price = base_price * (1 - self.sl)
            future_prices = prices[i+1:i+1+N_DAYS] if i+1+N_DAYS <= n else prices[i+1:]
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
        return targets

    def get_features(self):
        df = self.data.copy()
        for i in range(1, self.lookback+1):
            df[f"Close-{i}"] = df["Close"].shift(i)
        for period in self.ma_lens:
            df[f"SMA_{period}"] = ta.sma(df["Close"], length=period)
        vol_average = df["Volume"].rolling(window=20).mean().std()
        vol_change = (df["Volume"].shift(1) - df["Volume"]) / df["Volume"]
        df["VolChangeToAverage"] = vol_change / vol_average
        df["ATR"] = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = 0
        df.loc[macd['MACD_12_26_9'] > macd['MACDs_12_26_9'], 'MACD'] = 1
        df.drop(['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'], axis=1, inplace=True, errors='ignore')
        df['RSI'] = ta.rsi(df['Close'], length=14)
        if self.simple_target:
            df['Future_Return'] = (df['Close'].shift(-N_DAYS) - df['Close']) / df['Close']
            df["Target"] = 0
            df.loc[df['Future_Return'] > THRESHOLD, "Target"] = 1
        else:
            assert self.sl is not None
            df['Target'] = self.label_takeprofit_stoploss(df['Close'].values)
        return df.dropna()

    def train_model(self, test_size=0.2):
        self.data = self.get_features()
        x = self.data.select_dtypes(include=['float64', 'int64', 'int32']).drop(columns=['Target'], errors='ignore')
        y = self.data["Target"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)
        self.x_test = x_test
        self.pipeline.fit(x_train, y_train)
        features = x.columns
        y_pred = self.pipeline.predict(x_test)
        print("Accuracy score:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return {"pipeline": self.pipeline, "features": features, "data": self.data}

    def predict(self, x):
        return self.pipeline.predict(x)

    def simulate_trades(self):
        df = self.x_test.copy()
        x = df.select_dtypes(include=['float64', 'int64', 'int32']).drop(columns=['Target'], errors='ignore')
        df['Predicted'] = self.predict(x)
        
        initial_cash = 100000  
        position = 0  
        cash = initial_cash
        portfolio_values = []
        trades = []
        in_trade = False  
        entry_date = None
        entry_price = 0
        max_shares_per_trade = 100  
        
        for i in range(len(df)):
            portfolio_value = cash + position * df['Close'].iloc[i]
            portfolio_values.append(portfolio_value)

            if in_trade:
                current_price = df['Close'].iloc[i]
                days_held = (df.index[i] - entry_date).days

                if current_price >= entry_price * (1 + self.threshold): 
                  
                    sell_price = current_price * (1 - FEE - SLIPPAGE)  
                    cash += position * sell_price
                    trades.append({'Date': df.index[i], 'Type': 'Sell', 'Price': sell_price, 'Shares': position})
                    position = 0
                    in_trade = False
                    print(f"Sell (TP) at {df.index[i]}: Price={sell_price:.2f}, Shares={position}, Cash={cash:.2f}, Portfolio={portfolio_value:.2f}")
                elif current_price <= entry_price * (1 - self.sl):  
                    sell_price = current_price * (1 - FEE - SLIPPAGE)  
                    cash += position * sell_price
                    trades.append({'Date': df.index[i], 'Type': 'Sell', 'Price': sell_price, 'Shares': position})
                    position = 0
                    in_trade = False
                    print(f"Sell (SL) at {df.index[i]}: Price={sell_price:.2f}, Shares={position}, Cash={cash:.2f}, Portfolio={portfolio_value:.2f}")
                elif days_held >= N_DAYS:  
                    sell_price = current_price * (1 - FEE - SLIPPAGE) 
                    cash += position * sell_price
                    trades.append({'Date': df.index[i], 'Type': 'Sell', 'Price': sell_price, 'Shares': position})
                    position = 0
                    in_trade = False
                    print(f"Sell (N_DAYS) at {df.index[i]}: Price={sell_price:.2f}, Shares={position}, Cash={cash:.2f}, Portfolio={portfolio_value:.2f}")

            if not in_trade and df['Predicted'].iloc[i] == 1 and i < len(df) - N_DAYS: 
                entry_price = df['Close'].iloc[i] * (1 + FEE + SLIPPAGE) 
                shares = min(cash // entry_price, max_shares_per_trade) 
                if shares > 0:  
                    position = shares
                    cash -= shares * entry_price
                    entry_date = df.index[i]
                    trades.append({'Date': entry_date, 'Type': 'Buy', 'Price': entry_price, 'Shares': shares})
                    in_trade = True
                    print(f"Buy at {df.index[i]}: Price={entry_price:.2f}, Shares={shares}, Cash={cash:.2f}, Portfolio={portfolio_value:.2f}")
        
        if in_trade and i < len(df) - 1:
            current_price = df['Close'].iloc[i]
            sell_price = current_price * (1 - FEE - SLIPPAGE)  
            cash += position * sell_price
            trades.append({'Date': df.index[i], 'Type': 'Sell', 'Price': sell_price, 'Shares': position})
            position = 0
            in_trade = False
            print(f"Final Sell at {df.index[i]}: Price={sell_price:.2f}, Shares={position}, Cash={cash:.2f}, Portfolio={portfolio_value:.2f}")
        
        portfolio_values[-1] = cash + position * df['Close'].iloc[-1]
        df['Portfolio_Value'] = portfolio_values
        print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
        return df, trades

    def plot_trades_and_portfolio(self):
        df, trades = self.simulate_trades()
        
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['Close'], label='Close Price', color='blue')
        buy_signals = [t for t in trades if t['Type'] == 'Buy']
        sell_signals = [t for t in trades if t['Type'] == 'Sell']
        plt.scatter([t['Date'] for t in buy_signals], [t['Price'] for t in buy_signals], 
                    marker='^', color='green', label='Buy', s=100)
        plt.scatter([t['Date'] for t in sell_signals], [t['Price'] for t in sell_signals], 
                    marker='v', color='red', label='Sell', s=100)
        plt.title(f'{SYMBOL} Price with Trades')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['Portfolio_Value'], label='Portfolio Value', color='purple')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

model = DecisionTreeModel(
    lookback=LOOKBACK,
    symbol=SYMBOL,
    simple_target=False, 
    sl=SL, 
    threshold=THRESHOLD, 
    ma_lens=MA_LENS, 
    start=START, 
    end=END
)
output_data = model.train_model()
model.plot_trades_and_portfolio()
