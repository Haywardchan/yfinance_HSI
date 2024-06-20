import yfinance as yf
import pandas as pd
import numpy as np
import csv
import os
# # Testing Apple
# apple = yf.Ticker('AAPL')
# hist = apple.history(period = "1mo")
# print(hist)
# hist.to_csv('testing', index=False)

def output(df, filename, filetype='csv'):
    if filetype=='csv':
        fname=filename+'.csv'
        # print(len(df))
        data=np.array([df[i] for i in range(len(df))]).T
        with open(fname, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(list(data))
        # print(fname)
        # df.to_csv(filename+'.csv', index=False, encoding='utf-8-sig')
    elif filetype=='excel':
        fname=filename+'.xlsx'
        data=pd.DataFrame(np.array([df[i] for i in range(len(df))]).T)
        data.to_excel(fname)

def download_stock_data(output_file, stock_code, period="1mo"):
    """
    Downloads the stock data for the given stock code and saves it as a CSV file.

    Parameters:
    stock_code (str): The stock code to download data for.
    output_file (str): The path and filename to save the CSV file.
    period (str): The time period to download data for. Can be "max", "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", or "ytd". Default is "max".
    """
    # Download the stock data
    stock = yf.Ticker(stock_code)
    df = stock.history(period = period)
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the 'data' folder path
    data_folder = os.path.join(script_dir, 'data' + '_' + period)
    # Construct the output file path
    output_file = os.path.join(data_folder, output_file)
    # Create the 'data' folder if it doesn't exist
    os.makedirs(data_folder, exist_ok = True)
    # Save the dataframe as a CSV file
    df.to_csv(output_file)
    print(f"Stock data for {stock_code} ({period}) saved to {output_file}")

def read_stock_codes_from_file(filename):
    """
    Reads stock codes from a text file and stores them in a Python list.

    Parameters:
    filename (str): The name of the text file containing the stock codes.

    Returns:
    list: A list of stock codes.
    """
    stock_codes = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                stock_code = line.strip()
                if stock_code:
                    stock_codes.append(stock_code)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
    except IOError:
        print(f"Error: There was a problem reading the file '{filename}'.")
    return stock_codes

def download_stock_from_txt(file_txt, periods = ["1mo", "1y", "5y"]):
    tickers = read_stock_codes_from_file(file_txt)
    for code in tickers:
        for period in periods:
            download_stock_data(code + '_' + period + '.csv', code, period = period)

def preprocess_df(file_txt, periods = ["1mo", "1y", "5y"]):
    tickers = read_stock_codes_from_file(file_txt)
    for code in tickers:
        for period in periods:
            # Load the CSV file into a Pandas DataFrame
            df = pd.read_csv(f"data_{period}/{code}_{period}.csv")

            # Drop the days with only dividend
            df = df.dropna(subset=['Open'])

            # Calculate the return and add it as a new column
            df['Return (%)'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
            df.loc[0, 'Return (%)'] = 0

            # Save the updated DataFrame back to a CSV file
            df.to_csv(f'data_{period}/{code}_{period}.csv')
            print(f"Appended Stock Return data for {code} ({period}) saved to {file_txt}")

def calculate_stock_return(csv_file, start_date, end_date):
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
    df = pd.read_csv(csv_file)

    # Convert the date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter the data for the specified date range
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Calculate the return
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    return_pct = (end_price - start_price) / start_price

    return return_pct * 100

def calculate_risk_and_sharpe_ratio(file_txt, periods = ["1mo", "1y", "5y"]):
    risks, returns, sharpes, correlation = ["Risk"], ["ROI"], ["Sharpe Ratio"], ["Correlation to HSI"]
    tickers = read_stock_codes_from_file(file_txt)
    for period in periods:
        hsi = pd.read_csv(f"data_{period}/^HSI_{period}.csv")
        for code in tickers:
            # Load the CSV file into a Pandas DataFrame
            df = pd.read_csv(f"data_{period}/{code}_{period}.csv")
            # Calculate the overall Risk and Sharpe ratio of the stock
            # Risk of the stock
            stock_risk = df['Return (%)'].std() * np.sqrt(df.shape[0])
            # Sharpe ratio
            stock_TROI = calculate_stock_return(f"data_{period}/{code}_{period}.csv", '2023-06-12', '2024-06-11')
            stock_sharpe = stock_TROI / stock_risk 
            # Correlation to HSI
            # print(hsi['Return (%)'].shape, df['Return (%)'].shape, code)
            coeff = np.corrcoef(df['Return (%)'], hsi['Return (%)'])[0, 1]
            risks.append(stock_risk)
            returns.append(stock_TROI)
            sharpes.append(stock_sharpe)
            correlation.append(coeff)
        df = [["Stock"] + tickers, risks, returns, sharpes, correlation]
        output(df, f"HSI_{period}")
        print(f"exported HSI_{period}.csv")
        
def save_csv_to_postgreSQL(csvfile):
    import psycopg2
    conn = psycopg2.connect(host="localhost", dbname="finance", user='dev', password="93148325", port="5432")
    cur = conn.cursor()
    # Do sth
    cur.execute("""
    CREATE TABLE stock_performance (
        id SERIAL PRIMARY KEY,
        stock_name VARCHAR(50) NOT NULL,
        risk DECIMAL(10,2) NOT NULL,
        roi DECIMAL(10,2) NOT NULL,
        sharpe_ratio DECIMAL(10,2) NOT NULL,
        correlation_to_hsi DECIMAL(10,2) NOT NULL
    );
    """)
    # Open the CSV file and read the data
    with open(csvfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        # Insert the data into the PostgreSQL table
        for row in reader:
            stock_name, risk, roi, sharpe_ratio, correlation_to_hsi = row
            cur.execute("INSERT INTO stock_performance (stock_name, risk, roi, sharpe_ratio, correlation_to_hsi) VALUES (%s, %s, %s, %s, %s)", 
                    (stock_name, risk, roi, sharpe_ratio, correlation_to_hsi))

    # Commit the changes and close the connection
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    Not_downloaded = False
    stock_txt = 'HSI_stocks.txt'
    if Not_downloaded:
        download_stock_from_txt(stock_txt, ["1y"])
        preprocess_df(stock_txt, ["1y"])
        calculate_risk_and_sharpe_ratio(stock_txt, ["1y"])
    save_csv_to_postgreSQL('HSI_1y.csv')

            
