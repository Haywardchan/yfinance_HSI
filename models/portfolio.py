import pandas as pd
import numpy as np
from scipy.stats import norm
from .stock import Stock

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
    def rebalanced_returns(self, days_to_rebalance = 1):
        df = self.rebalanced_prices(100, days_to_rebalance)
        # Calculate the return and add it as a new column
        returns = (df['Rebalanced Prices'] / df['Rebalanced Prices'].shift(1) - 1) * 100
        returns[0] = 0
        return pd.DataFrame({'Rebalanced Portfolio Returns': returns})
    
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
    
    def prices(self, init_price, period="1y"):
        self.period = period
        portfolio_prices = [init_price]
        stock_prices = [weight * init_price for weight in self.portions]
        merged_df = self.merged_returns()[1:]
        for idx in range(len(merged_df)):
            stock_prices = np.multiply(1 + merged_df.iloc[idx].values / 100, stock_prices)
            portfolio_prices.append(np.sum(stock_prices, axis = 0))
        # print(portfolio_prices, len(portfolio_prices))
        return pd.DataFrame({'Portfolio Prices': portfolio_prices[1:]}, index=self.get_dates()[1:])

    def rebalanced_prices(self, init_price, days_to_rebalance = 1, period="1y"):
        self.period=period
        all_stock_prices = [init_price]
        stock_prices = [weight * init_price for weight in self.portions]
        # print(stock_prices)
        merged_df = self.merged_returns()[1:]
        # print(merged_df)
        for idx in range(len(merged_df)):
            stock_prices = np.multiply(1 + merged_df.iloc[idx].values / 100, stock_prices)
            # print("Returns", merged_df.iloc[idx].values)
            # print("Stock Prices", stock_prices)
            if idx % days_to_rebalance == 0:
                # Rebalance the prices
                row_sum = np.sum(stock_prices, axis = 0)
                stock_prices = [weight * row_sum for weight in self.portions]
                # print("rebalanced Stock Prices", stock_prices)
            all_stock_prices.append(np.sum(stock_prices, axis = 0))
        # print(all_stock_prices, len(all_stock_prices))
        return pd.DataFrame({'Rebalanced Prices': all_stock_prices}, index=self.get_dates())

    def covariance_matrix(self):
        merged_returns = self.merged_returns()
        return (merged_returns / 100).cov().values * merged_returns.shape[0] 
            
    def merged_returns(self):
        returns = []
        for i, stock in enumerate(self.stocks):
            df = pd.read_csv(f"data_{self.period}/{stock.stock_id}_{self.period}.csv")
            returns.append(df['Return (%)'])
        merged_df = pd.concat(returns, axis=1)
        return merged_df.fillna(0)  # fill nan as 0 for an assumption
    
    def get_dates(self):
        """
        Returns the dates from the stock data CSV file for the given period.

        Returns:
        list: A list of dates as strings.
        """
        df = pd.read_csv(f"data_5y/{self.stocks[0].stock_id}_5y.csv")
        dates = df['Date'].tolist()
        merged_df = self.merged_returns()
        
        # Remove dates where merged_df has NaN values
        valid_dates = [date for date, value in zip(dates, merged_df.notna().all(axis=1)) if value]
        
        if len(valid_dates) != merged_df.shape[0]:
            raise ValueError("The number of valid dates must match the number of merged returns.")
        
        return valid_dates
    
    def __str__(self):
        portfolio_info = "Portfolio Information:\n"
        for i, stock in enumerate(self.stocks):
            portfolio_info += f"{stock.stock_name}: {self.portions[i] * 100}%\n"
        portfolio_info += f"\nRisk: {self.risk:.2f}\nROI: {self.roi:.2f}\nSharpe Ratio: {self.sharpe_ratio:.2f}\nVaR: {self.var:.2f}\nBeta: {self.beta:.2f}"
        return portfolio_info

# x = Portfolio()
# stockA = Stock("0001.HK")
# stockB = Stock("0005.HK")
# stockC = Stock("0002.HK")
# x.add_stock(stockA, 0.3)
# x.add_stock(stockB, 0.3)
# x.add_stock(stockC, 0.4)
# # x.rebalanced_prices(100, 1)
# print(x.prices(100))