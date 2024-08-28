import yfinance as yf
from scipy.stats import norm
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
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    
    if period not in valid_periods:
        print("Invalid period specified, searching for larger dataset...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # create a new folder called data_3y
        data_3y_folder = os.path.join(script_dir, 'data_3y')
        os.makedirs(data_3y_folder, exist_ok=True)

        # Check if the specific file exists in the data_5y folder
        data_5y_folder = os.path.join(script_dir, 'data_5y')
        specific_filename = f"{stock_code}_5y.csv"
        file_path = os.path.join(data_5y_folder, specific_filename)
        
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            # Slice the dataframe to get the last 3 years of data
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            max_date = df['Date'].max()
            three_years_ago = max_date - pd.DateOffset(years=3)
            df_sliced = df[df['Date'] >= three_years_ago]
            # Save the dataframe as a new csv file in the created data_3y folder with the new filename
            new_filename = specific_filename.replace('_5y', '_3y')
            new_file_path = os.path.join(data_3y_folder, new_filename)
            df_sliced.to_csv(new_file_path, index=False)
            print(f"Stock data for {stock_code} ({period}) saved to {new_file_path}")
        else:
            print(f"No data found for {stock_code} in the data_5y folder.")
    else:
        # Download the stock data
        stock = yf.Ticker(stock_code)
        df = stock.history(period=period)
        # Get the directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the 'data' folder path
        data_folder = os.path.join(script_dir, 'data' + '_' + period)
        # Construct the output file path
        output_file = os.path.join(data_folder, output_file)
        # Create the 'data' folder if it doesn't exist
        os.makedirs(data_folder, exist_ok=True)
        # Save the dataframe as a CSV file
        df.to_csv(output_file)
        print(f"Stock data for {stock_code} ({period}) saved to {output_file}")

def plot_price_graph(output_file, stock_code, period="1mo"):
    import pandas as pd
    import matplotlib.pyplot as plt
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the 'data' folder path
    data_folder = os.path.join(script_dir, 'charts' + '_' + period)
    # Construct the output file path
    output_file = os.path.join(data_folder, output_file)
    # Create the 'charts' folder if it doesn't exist
    os.makedirs(data_folder, exist_ok = True)
    df = pd.read_csv(f'data_{period}/{stock_code}_{period}.csv')
    # Plot the close price
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'])
    # plt.plot(df['Date'], df['Return (%)'], color="blue")
    plt.title('Stock Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Stock price chart for {stock_code} ({period}) saved to {output_file}")

def plot_volume_graph(output_file, stock_code, period="1mo"):
    import pandas as pd
    import matplotlib.pyplot as plt
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the 'data' folder path
    data_folder = os.path.join(script_dir, 'charts' + '_' + period)
    # Construct the output file path
    output_file = os.path.join(data_folder, output_file)
    # Create the 'charts' folder if it doesn't exist
    os.makedirs(data_folder, exist_ok = True)
    df = pd.read_csv(f'data_{period}/{stock_code}_{period}.csv')
    # Extract the necessary data
    stock_dates = df['Date']
    stock_volume = df['Volume']

    # Plot the volume
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(stock_dates, stock_volume, color='blue')
    ax.set_title(f'Trading Volume for {stock_code}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Stock price chart for {stock_code} ({period}) saved to {output_file}")

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

def plot_graphs_from_txt(file_txt, periods = ["1mo", "1y", "5y"]):
    tickers = read_stock_codes_from_file(file_txt)
    for code in tickers:
        for period in periods:
            plot_price_graph('P' + code + '_' + period + '.png', code, period)
            plot_volume_graph('V' + code + '_' + period + '.png', code, period)

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

def calculate_stock_return(csv_file):
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

    # Calculate the return
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    return_pct = (end_price - start_price) / start_price

    return return_pct * 100

def calculate_KPIs(file_txt, periods = ["1mo", "1y", "5y"], index="HSI"):
    risks, returns, sharpes, correlation, pe, VaR, names, price_charts, volume_charts = ["risk"], ["roi"], ["sharpe_ratio"], ["correlation_to_hsi"], ["PE_ratio"], ["var"], ["stock_name"], [], []
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
                stock_TROI = calculate_stock_return(f"data_{period}/{code}_{period}.csv")
            except:
                stock_TROI = 0
                print(f"TROI cannot be calculated for {code}")
            stock_sharpe = stock_TROI / stock_risk 
            # Correlation to HSI
            try:
                common_dates = pd.DataFrame({'Date': df['Date'].unique()})
                common_dates = common_dates[common_dates['Date'].isin(hsi['Date'].unique())]
                coeff = np.corrcoef(df[df['Date'].isin(common_dates['Date'])]['Return (%)'], hsi[hsi['Date'].isin(common_dates['Date'])]['Return (%)'])[0, 1]
            except:
                print(hsi['Return (%)'].shape, df['Return (%)'].shape, code)
            # P/E ratio
            stk_info = yf.Ticker(code).info
            try:
                stock_PE_ratio = round(stk_info["currentPrice"] / stk_info["trailingEps"], 2)
            except:
                stock_PE_ratio = 0
                print(f"key error occurs in {code}")
            # 95% VaR
            stock_VaR = norm.ppf(0.05) * df['Return (%)'].std() * np.sqrt(365/252)
            # Stock Name
            stkname = stk_info["longName"]
            price_trend_path = f"charts_{period}/P{code}_{period}.png"
            volume_graph_path = f"charts_{period}/V{code}_{period}.png"
            
            if os.path.exists(price_trend_path):
                with open(price_trend_path, 'rb') as f:
                    binary_data = f.read()
                price_charts.append(binary_data)
            else:
                print("Price trend chart does not exist")

            if os.path.exists(volume_graph_path):
                with open(volume_graph_path, 'rb') as f:
                    binary_data = f.read()
                volume_charts.append(binary_data)
            else:
                print("Volume chart does not exist")
            risks.append(stock_risk)
            returns.append(stock_TROI)
            sharpes.append(stock_sharpe)
            correlation.append(coeff)
            pe.append(stock_PE_ratio)
            VaR.append(stock_VaR)
            names.append(stkname)
            print(f"KPI of {code} is calculated")
        df = [["stock_id"] + tickers, names, risks, returns, sharpes, correlation, pe, VaR]
        # save_df_to_postgresql(lists_to_df(df[:][1:]), "finance", "stock_performance")
        output(df, f"{index}_{period}")
        print(f"exported {index}_{period}.csv")
        return [price_charts, volume_charts]
        
def save_csv_to_postgreSQL(csvfile, table_name = "stock_performance", assets = []):
    import psycopg2
    conn = psycopg2.connect(host="localhost", dbname="finance", user='postgres', password="1234", port="5432")
    cur = conn.cursor()
    # Do sth
    cur.execute("""ROLLBACK""")
    cur.execute(f"""
    DROP TABLE IF EXISTS {table_name}
    """)
    # cur.execute(f"""
    # CREATE TABLE {table_name} (
    #     id SERIAL PRIMARY KEY,
    #     stock_id VARCHAR(50) NOT NULL,
    #     stock_name VARCHAR(100) NOT NULL,
    #     risk DECIMAL(10,2) NOT NULL,
    #     roi DECIMAL(10,2) NOT NULL,
    #     sharpe_ratio DECIMAL(10,2) NOT NULL,
    #     correlation_to_hsi DECIMAL(10,2) NOT NULL,
    #     PE_ratio DECIMAL(10,2) NOT NULL,
    #     VaR DECIMAL(10,2) NOT NULL,
    #     price_chart BYTEA NOT NULL,
    #     volume_chart BYTEA NOT NULL
    # );
    # """)
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
    # Open the CSV file and read the data
    with open(csvfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        # Insert the data into the PostgreSQL table
        for idx, row in enumerate(reader):
            stock_id, stock_name, risk, roi, sharpe_ratio, correlation_to_hsi, PE_ratio, VaR = row
            # cur.execute(f"INSERT INTO {table_name} (stock_id, stock_name, risk, roi, sharpe_ratio, correlation_to_hsi, PE_ratio, VaR, price_chart, volume_chart) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", 
            #         (stock_id, stock_name, risk, roi, sharpe_ratio, correlation_to_hsi, PE_ratio, VaR, assets[0][idx], assets[1][idx]))
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

def csv_to_json(filename, filepath):
    df = pd.read_csv(filepath + '/' + filename)
    # df = pd.read_csv('data_1y/' + filename + '.csv')
    # df.to_json('json_data/' + filename.replace(".csv", "") + '.json', indent = 1)
    df.to_json('json_data/' + filename.replace(".csv", "") + '.json', indent = 1, orient = 'records')
    print(f"{filename} is converted to json")

def convert_data_to_json(folder_path):
    # Get a list of all items (files and directories) in the folder
    items = os.listdir(folder_path)

    # Iterate through the items in the folder
    for item in items:
        csv_to_json(item, folder_path)

def hsi():
    Not_downloaded = True
    stock_txt = 'HSI_stocks.txt'
    if Not_downloaded:
        download_stock_from_txt(stock_txt, ["1y"])
        plot_graphs_from_txt(stock_txt, ["1y"])
        preprocess_df(stock_txt, ["1y"])
        graphs = calculate_KPIs(stock_txt, ["1y"])
    save_csv_to_postgreSQL('HSI_1y.csv', "stock_performance", graphs)

def sse():
    Not_downloaded = True
    stock_txt = 'A_stocks.txt'
    if Not_downloaded:
        download_stock_from_txt(stock_txt, ["1y"])
        plot_graphs_from_txt(stock_txt, ["1y"])
        preprocess_df(stock_txt, ["1y"])
        calculate_KPIs(stock_txt, ["1y"], "000001.SS")
    save_csv_to_postgreSQL('000001.SS_1y.csv', "stock_performance_sse")

def nasdaq():
    Not_downloaded = True
    stock_txt = 'NASDAQ_stocks.txt'
    if Not_downloaded:
        download_stock_from_txt(stock_txt, ["1y"])
        plot_graphs_from_txt(stock_txt, ["1y"])
        preprocess_df(stock_txt, ["1y"])
        graphs = calculate_KPIs(stock_txt, ["1y"], "^NDX")
    save_csv_to_postgreSQL('^NDX_1y.csv', "stock_performance_nasdaq", graphs)

def analyze_index(index, stock_txt, db_table_name, periods):
    Not_downloaded = True
    if Not_downloaded:
        download_stock_from_txt(stock_txt, periods)
        # plot_graphs_from_txt(stock_txt, periods)
        preprocess_df(stock_txt, periods)
        graphs = calculate_KPIs(stock_txt, periods, index)
    for period in periods:
        table_name = db_table_name + "_" + period
        save_csv_to_postgreSQL(f'{index}_{period}.csv', table_name)

if __name__ == "__main__":
    # analyze_index("^NDX", 'NASDAQ_stocks.txt', "stock_performance_nasdaq", ["1y","2y","5y"])
    # analyze_index("^HSI", 'HSI_stocks.txt', "stock_performance_hsi", ["1y","2y","5y"])
    # analyze_index("000001.SS", 'A_stocks.txt', "stock_performance_sse", ["1y","2y","5y"])
    # # os.mkdir('json_data')
    # convert_data_to_json('data_1y')
    # convert_data_to_json('data_2y')
    # convert_data_to_json('data_5y')
    analyze_index("^NDX", 'NASDAQ_stocks.txt', "stock_performance_nasdaq", ["10y"])
    analyze_index("^HSI", 'HSI_stocks.txt', "stock_performance_hsi", ["10y"])
    analyze_index("000001.SS", 'A_stocks.txt', "stock_performance_sse", ["10y"])
    convert_data_to_json('data_10y')



            
