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

def calculate_KPIs(file_txt, periods = ["1mo", "1y", "5y"], index="HSI"):
    risks, returns, sharpes, correlation, pe, VaR, names = ["risk"], ["roi"], ["sharpe_ratio"], ["correlation_to_hsi"], ["PE_ratio"], ["var"], ["stock_name"]
    tickers = read_stock_codes_from_file(file_txt)
    for period in periods:
        if index == "HSI":
            hsi = pd.read_csv(f"data_{period}/^HSI_{period}.csv")
        else:
            hsi = pd.read_csv(f"data_{period}/{index}_{period}.csv")
        for code in tickers:
            # Load the CSV file into a Pandas DataFrame
            df = pd.read_csv(f"data_{period}/{code}_{period}.csv")
            # Calculate the overall Risk and Sharpe ratio of the stock
            # Risk of the stock
            stock_risk = df['Return (%)'].std() * np.sqrt(df.shape[0])
            # Sharpe ratio
            try:
                stock_TROI = calculate_stock_return(f"data_{period}/{code}_{period}.csv", '2023-06-12', '2024-06-11')
            except:
                stock_TROI = 0
                print(f"TROI cannot be calculated for {code}")
            stock_sharpe = stock_TROI / stock_risk 
            # Correlation to HSI
            # print(hsi['Return (%)'].shape, df['Return (%)'].shape, code)
            coeff = np.corrcoef(df['Return (%)'], hsi['Return (%)'])[0, 1]
            # P/E ratio
            stk_info = yf.Ticker(code).info
            try:
                stock_PE_ratio = round(stk_info["currentPrice"] / stk_info["trailingEps"], 2)
            except:
                stock_PE_ratio = 0
                print(f"key error occurs in {code}")
            # 95% VaR
            stock_VaR = df['Return (%)'].std() * 1.65
            # Stock Name
            stkname = stk_info["longName"]
            risks.append(stock_risk)
            returns.append(stock_TROI)
            sharpes.append(stock_sharpe)
            correlation.append(coeff)
            pe.append(stock_PE_ratio)
            VaR.append(stock_VaR)
            names.append(stkname)
            print(f"KPI of {code} is calculated")
        df = [["stock_id"] + tickers, names, risks, returns, sharpes, correlation, pe, VaR]
        save_df_to_postgresql(lists_to_df(df[:][1:]), "finance", "stock_performance")
        output(df, f"{index}_{period}")
        print(f"exported {index}_{period}.csv")
        
def save_csv_to_postgreSQL(csvfile, table_name = "stock_performance"):
    import psycopg2
    conn = psycopg2.connect(host="localhost", dbname="finance", user='dev', password="93148325", port="5432")
    cur = conn.cursor()
    # Do sth
    try:
        # cur.execute("""ROLLBACK""")
        cur.execute(f"""
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            stock_id VARCHAR(50) NOT NULL,
            stock_name VARCHAR(100) NOT NULL,
            risk DECIMAL(10,2) NOT NULL,
            roi DECIMAL(10,2) NOT NULL,
            sharpe_ratio DECIMAL(10,2) NOT NULL,
            correlation_to_hsi DECIMAL(10,2) NOT NULL,
            PE_ratio DECIMAL(10,2) NOT NULL,
            VaR DECIMAL(10,2) NOT NULL
        );
        """)
    except:
        print('The table exists')
    # Open the CSV file and read the data
    with open(csvfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        # Insert the data into the PostgreSQL table
        for row in reader:
            stock_id, stock_name, risk, roi, sharpe_ratio, correlation_to_hsi, PE_ratio, VaR = row
            cur.execute(f"INSERT INTO {table_name} (stock_id, stock_name, risk, roi, sharpe_ratio, correlation_to_hsi, PE_ratio, VaR) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", 
                    (stock_id, stock_name, risk, roi, sharpe_ratio, correlation_to_hsi, PE_ratio, VaR))

    # Commit the changes and close the connection
    conn.commit()
    cur.close()
    conn.close()
    print(f"{csvfile} is saved to table: {table_name} in postgreSQL")

def save_df_to_postgresql(df, database_name, table_name = "finance", user='postgres', password='1234'):
    from sqlalchemy import create_engine
    """
    Saves a Pandas DataFrame to a PostgreSQL database.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be saved.
    table_name (str): The name of the table to be created/updated.
    user (str): The username for the PostgreSQL database (default is 'your_username').
    password (str): The password for the PostgreSQL database (default is 'your_password').
    """
    # Create a SQLAlchemy engine to connect to the PostgreSQL database
    engine = create_engine(f'postgresql://{user}:{password}@localhost:5432/{database_name}')

    # Write the DataFrame to the PostgreSQL database
    df.iloc[1:].to_sql(table_name, engine, if_exists='replace', index=False)

    print(f"DataFrame saved to PostgreSQL table '{table_name}' in {database_name}.")

def lists_to_df(lists):
    list_transposed = list(zip(*lists))
    return pd.DataFrame(list_transposed, columns=list_transposed[0])

def hsi():
    Not_downloaded = False
    stock_txt = 'HSI_stocks.txt'
    if Not_downloaded:
        download_stock_from_txt(stock_txt, ["1y"])
        preprocess_df(stock_txt, ["1y"])
        calculate_KPIs(stock_txt, ["1y"])
    save_csv_to_postgreSQL('HSI_1y.csv')

def sse():
    Not_downloaded = True
    stock_txt = 'A_stocks.txt'
    # if Not_downloaded:
        # download_stock_from_txt(stock_txt, ["1y"])
        # preprocess_df(stock_txt, ["1y"])
        # calculate_KPIs(stock_txt, ["1y"], "000001.SS")
    save_csv_to_postgreSQL('000001.SS_1y.csv', "stock_performance_sse")

if __name__ == "__main__":
    hsi()


            
