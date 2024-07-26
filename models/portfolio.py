import pandas as pd
import numpy as np
from scipy.stats import norm
from stock import Stock

class Portfolio:
    def __init__(self):
        self.stocks = []
        self.portions = []
        self.period = "1y"
     
    def set_portfolio(self, stocks, portions, period = "1y"):
        self.stocks = stocks
        self.portions = portions
        self.period = period

    def add_stock(self, stock, portion):
        if(not (portion <= 0 or portion > 1)):
            self.stocks.append(stock)
            self.portions.append(portion)
    
    def clear_stock(self):
        self.stocks = []
        self.portions = []

    def remove_stock(self, index):
        # delete the stock and weights
        del self.stocks[index]
        del self.portions[index]
        # renormalize the weights
        self.portions = self.portions/np.sum(self.portions)
    
    def get_stocks(self):
        return self.stocks
    
    def get_weights(self):
        return self.portions

    def set_weights(self, weights):
        assert(len(self.stocks)==len(weights))
        self.portions = weights/ np.sum(weights)
        return self

    def filter_stock_by_sharpe_ratio(self, num_stocks = 20):
        sorted_s, sorted_p = zip(*sorted(zip(self.stocks, self.portions), key= lambda x: x[0].sharpe_ratio, reverse=True))
        filtered_stocks = sorted_s[:num_stocks]
        filtered_portion = sorted_p[:num_stocks]
        self.set_portfolio(filtered_stocks, filtered_portion)
        return self

    def isEmpty(self) -> bool:
        assert(len(self.stocks)==len(self.portions))
        return len(self.stocks) == 0 
    
    def normalize_weights(self):
        self.portions = self.portions/np.sum(self.portions)

    @property
    def risk(self):
        if(self.isEmpty()):
            return 0
        return np.sqrt(np.dot(np.array(self.portions).T, np.dot(self.covariance_matrix(), np.array(self.portions)))) * 100

    @property
    def daily_returns(self):
        return np.dot(self.merged_returns(), np.array(self.portions))

    @property
    def roi(self):
        if(self.isEmpty()):
            return 0
        return np.dot(np.array([stock.calculate_stock_return() for stock in self.stocks]), np.array(self.portions)).sum()

    @property
    def sharpe_ratio(self):
        if(self.isEmpty()):
            return 0
        return self.roi / self.risk

    @property
    def var(self):
        if(self.isEmpty()):
            return 0
        return norm.ppf(0.05) * self.daily_returns.std() * np.sqrt(365/252)

    @property
    def beta(self):
        if(self.isEmpty()):
            return 0
        return self.risk / Stock("^HSI").risk
    
    # Given a portfolio return a return matrix for everyday with rebalancing
    @property
    def rebalanced_returns(self):
        return 
    
    # Given a portfolio return a return matrix for everyday without rebalancing
    @property
    def returns(self):
        # Check that the number of weights matches the number of columns in the returns DataFrame
        merged_df = self.merged_returns()
        if len(self.portions) != merged_df.shape[1]:
            raise ValueError("The number of weights must match the number of columns in the returns DataFrame.")
        
        # Calculate the weighted returns for each stock
        weighted_returns = merged_df.mul(self.portions, axis=1)
        
        # Calculate the combined returns by summing the weighted returns
        portfolio_returns = weighted_returns.sum(axis=1)
        
        # Create a new DataFrame with the combined returns
        portfolio_returns_df = pd.DataFrame({'Portfolio Returns': portfolio_returns})
        
        return portfolio_returns_df

    def covariance_matrix(self):
        merged_returns = self.merged_returns()
        return (merged_returns / 100).cov().values * merged_returns.shape[0] 
            
    def merged_returns(self):
        returns = []
        for i, stock in enumerate(self.stocks):
            df = pd.read_csv(f"data_{self.period}/{stock.stock_id}_{self.period}.csv")
            returns.append(df['Return (%)'])
            merged_df = pd.concat(returns, axis=1)
        return merged_df.fillna(0) # fill nan as 0 for an assumption
    
    def __str__(self):
        portfolio_info = "Portfolio Information:\n"
        for i, stock in enumerate(self.stocks):
            portfolio_info += f"{stock.stock_name}: {self.portions[i] * 100}%\n"
        portfolio_info += f"\nRisk: {self.risk:.2f}\nROI: {self.roi:.2f}\nSharpe Ratio: {self.sharpe_ratio:.2f}\nVaR: {self.var:.2f}\nBeta: {self.beta:.2f}"
        return portfolio_info

x = Portfolio()
stockA = Stock("0001.HK")
stockB = Stock("0005.HK")
stockC = Stock("0002.HK")
x.add_stock(stockA, 0.3)
x.add_stock(stockB, 0.3)
x.add_stock(stockC, 0.4)
print(x.merged_returns())
print(x.returns)