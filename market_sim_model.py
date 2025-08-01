import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
class Utils:
    @staticmethod
    def get_data_yahoo(stocks, start, end):
        data = yf.download(stocks, start=start, end=end, interval="1d")["Close"]
        print("get_data_yahoo")
        return data
    
    @staticmethod
    def get_data(stocks, start=None, end=None, n_days=None):
        print("get_data")
        n_days = n_days if n_days is not None else 300
        end = dt.datetime.now() if end is None else end
        start = end - dt.timedelta(days = 300) if start is None else start
        
        stock_prices = Utils.get_data_yahoo(stocks, start, end)
        returns = stock_prices.pct_change().dropna()
        
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        return (mean_returns, cov_matrix)
    

class MS(Utils):
    def __init__(self):
        print("__init__")
        
        
    def __call__(self, stocks, start=None, end=None, seed=None, sims=100, T=100, portfolio_val = 10_000, dt=1, sim_type="gbm"):
        print("__call__")
        if seed is not None:
            np.random.seed(seed)
        
        self.sim_type = sim_type
        self.dt = dt
        self.T = T
        self.sims = sims
        
        (self.mean_returns, self.cov_matrix) = Utils.get_data(stocks=stocks, start=start, end=end)
        
        self.weights = np.random.random(len(self.mean_returns))
        self.weights /= np.sum(self.weights)
        
        self.mean_mtx = np.full(shape=(self.T, len(self.weights)), fill_value=self.mean_returns)
        self.mean_mtx = self.mean_mtx.T
        
        self.portfolio_sims = np.full(shape=(self.T, self.sims), fill_value=0.0)
        self.init_portfolio = portfolio_val
        
        del sims, portfolio_val, T, start, end, stocks, dt
        
        match self.sim_type:
            case "gbm":
                run_simulation = self.run_gbm_sim
                
            case "mc":
                run_simulation = self.run_mc_sim
                
            case _:
                raise Exception(f"Simulation type {self.sim_type} is not available")
            
        run_simulation()
        print(f"VaR (Value at Risk): {self.get_var()}")
        self.plot()
            
    def run_mc_sim(self):
        L = np.linalg.cholesky(self.cov_matrix)
        
        for m in range(self.sims):
            Z = np.random.normal(size=(self.T, len(self.weights)))
            correlated_shock = Z @ L.T 
            daily_returns = self.mean_returns.values + correlated_shock
            weighted_returns = daily_returns @ self.weights
            self.portfolio_sims[:, m] = self.init_portfolio * np.cumprod(1 + weighted_returns)

    def run_gbm_sim(self):        
        for m in range(self.sims):
            mu = self.mean_returns.values
            sigma = np.sqrt(np.diag(self.cov_matrix))
            Z = np.random.normal(size=(self.T, len(self.weights)))
            drift = (mu - 0.5 * sigma ** 2) * self.dt
            shock = Z * sigma * np.sqrt(self.dt)
            daily_returns = np.exp(drift + shock)
            
            weighted_returns = daily_returns @ self.weights 
            self.portfolio_sims[:, m] = self.init_portfolio * np.cumprod(weighted_returns)
            
    def get_var(self, confidence=0.05):
        ending_values = self.portfolio_sims[-1, :]
        var = np.percentile(ending_values, 100 * confidence)
        return var
    
    def plot(self):
        plt.plot(self.portfolio_sims)
        plt.ylabel("Portfolio Val")
        plt.xlabel("Days")
        plt.title("MC sim")
        plt.show()
        
        
if __name__ == "__main__":
    stock_list = ["BTC-USD", "ETH-USD", "XRP-USD"]
    market_sim = MS()
    market_sim(stock_list,sim_type="mc")
    
