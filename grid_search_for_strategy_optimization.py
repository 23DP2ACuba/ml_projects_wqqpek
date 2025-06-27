from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas as pd
import yfinance as yf

class IchimokuCloudStrategy(Strategy):
    tenkan_period = 7
    kijun_period = 20
    senkou_period = 51
    
    def init(self):
        self.tenkan_sen = self.I(
            lambda x: talib.SMA(x, timeperiod=self.tenkan_period), 
            self.data.Close
        )
        self.kijun_sen = self.I(
            lambda x: talib.SMA(x, timeperiod=self.kijun_period),
            self.data.Close
        )
        
        self.senkou_a = self.I(
            lambda x: ((talib.SMA(x, timeperiod=self.tenkan_period) + 
                       talib.SMA(x, timeperiod=self.kijun_period)) / 2),
            self.data.Close
        )
        
        self.senkou_b = self.I(
            lambda x: talib.SMA(x, timeperiod=self.senkou_period),
            self.data.Close
        )
        
        self.chikou_span = self.I(
            lambda x: pd.Series(x).shift(-self.kijun_period).values,
            self.data.Close
        )

    def next(self):
        price = self.data.Close[-1]
        cloud_top = max(self.senkou_a[-1], self.senkou_b[-1])
        cloud_bottom = min(self.senkou_a[-1], self.senkou_b[-1])
        
        if (price > cloud_top and 
            self.tenkan_sen[-1] > self.kijun_sen[-1] and 
            not self.position):
            self.buy()
            
        elif (price < cloud_bottom and 
              not self.position):
            self.sell()
            
        if self.position.is_long and price < cloud_bottom:
            self.position.close()
        elif self.position.is_short and price > cloud_top:
            self.position.close()

class GridSearch():
    def __init__(self, start, end, start_val, end_val, ticker):
        self.start = start
        self.end = end
        self.start_val = start_val
        self.end_val = end_val
        self.ticker = ticker

    def run(self, tenkan_period = None, kijun_period = None, senkou_period = None, train: bool = False):
        if train == True:
            return self.bt.optimize(
                tenkan_period=range(7, 15),
                kijun_period=range(20, 35, 5),
                senkou_period=range(45, 65),
                maximize='Equity Final [$]',
                constraint=lambda p: p.tenkan_period < p.kijun_period < p.senkou_period
            )
        else:
            return self.bt.run(
                tenkan_period=int(tenkan_period),
                kijun_period=int(kijun_period),
                senkou_period=int(senkou_period)
            )
       

    def optim(self, plot=False):

        for train in [True, False]:
            if train == False:
                print("testing strategy>>")
                self.start = self.start_val
                self.end = self.end_val
            
            else:
                print("Optimizing strategy")

            self.data = yf.Ticker(self.ticker).history(start=self.start, end=self.end)
            self.data.drop(columns=["Stock Splits", "Dividends"], inplace=True, errors='ignore')
            self.data = self.data.dropna()

            self.bt = Backtest(
                self.data,
                IchimokuCloudStrategy,
                cash=100000,
                commission=0.002,
                exclusive_orders=True
            )

            if train == True:
                stats = self.run(train=train)
                self.tenkan_period = stats._strategy.tenkan_period
                self.kijun_period = stats._strategy.kijun_period
                self.senkou_period = stats._strategy.senkou_period

            elif train == False:
                stats = self.run(train=train,
                                 tenkan_period = self.tenkan_period,
                                 kijun_period = self.kijun_period,
                                 senkou_period = self.senkou_period
                                )

            else:
                break

            print(stats)
            
        if plot == True:
           self.bt.plot()

if __name__ == "__main__":
    strat = GridSearch(
        start = "2020-01-01",
        end = "2024-06-16",
        start_val = "2024-06-16",
        end_val = "2025-06-27",
        ticker = "BTC"
    )
    strat.optim()
