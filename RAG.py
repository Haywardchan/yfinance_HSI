import yfinance as yf
import pandas as pd
# print(pd.DataFrame.from_dict(yf.Ticker('^HSI').info, orient='index').reset_index().to_string())
# print(yf.Ticker('^HSI').info["currentPrice"])
# print(yf.Ticker("2388.HK").info)

import pandas as pd
import matplotlib.pyplot as plt

# # Assuming the CSV file is named 'stock_data.csv'
df = pd.read_csv('data_1y/2388.HK_1y.csv')
# df2 = pd.read_csv('data_1y/^HSI_1y.csv')
# # Plot the close price
# plt.figure(figsize=(12, 6))
# # plt.plot(df['Date'], df['Close'])
# plt.plot(df['Date'], df['Return (%)'], color="blue")
# plt.title('Day Return Trend')
# plt.xlabel('Date')
# plt.ylabel('Return')
# plt.grid(True)
# plt.legend()
# plt.show()

# Extract the necessary data
stock_dates = df['Date']
stock_volume = df['Volume']

# Plot the volume
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(stock_dates, stock_volume, color='blue')
ax.set_title('Trading Volume for 2388.HK')
ax.set_xlabel('Date')
ax.set_ylabel('Volume')
ax.grid(True)

plt.tight_layout()
plt.show()