import pandas as pd
class Stock:
    def __init__(self, stock_id, index="hsi"):
        self.period = "1y"
        if index == "hsi":
            df = pd.read_csv(f"HSI_{self.period}.csv")
        elif index == "sse":
            df = pd.read_csv(f"000001.SS_{self.period}.csv")
        elif index == "nasdaq":
            df = pd.read_csv(f"^NDX_{self.period}.csv")
        row = df.loc[df['stock_id']==stock_id]
        self.stock_id = stock_id
        # We cannot change the schema of the csv
        self.stock_name = row.iloc[0, 1]
        self.risk = row.iloc[0, 2]
        self.roi = row.iloc[0, 3]
        self.sharpe_ratio = row.iloc[0, 4]
        self.correlation_to_hsi = row.iloc[0, 5]
        self.PE_ratio = row.iloc[0, 6]
        self.VaR = row.iloc[0, 7]

    def calculate_stock_return(self):
        # print(calculate_stock_return('data_1y/0001.HK_1y.csv', '2023-06-12', '2024-06-11'))
        """
        Calculates the return of a stock based on the stock data in a CSV file.

        Parameters:
        csv_file (str): The path to the CSV file containing the stock data.
        start_date (str): The start date for the return calculation in the format 'YYYY-MM-DD'.
        end_date (str): The end date for the return calculation in the format 'YYYY-MM-DD'.

        Returns:
        float: The calculated return of the stock.
        """
        # Load the stock data from the CSV file
        df = pd.read_csv(f"data_{self.period}/{self.stock_id}_{self.period}.csv")

        # Calculate the return
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        return_pct = (end_price - start_price) / start_price

        return return_pct * 100

    def __str__(self):
        return f"Stock ID: {self.stock_id}\nStock Name: {self.stock_name}\nRisk: {self.risk}\nROI: {self.roi}\nSharpe Ratio: {self.sharpe_ratio}\nCorrelation to HSI: {self.correlation_to_hsi}\nP/E Ratio: {self.PE_ratio}\nVaR: {self.VaR}"
    
# print(Stock("0005.HK"))