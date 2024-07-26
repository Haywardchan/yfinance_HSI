import random
import pandas as pd
import numpy as np
from models import Stock, Portfolio, Asset_allocator
import matplotlib.pyplot as plt
def generated_portfolio(solution) -> Portfolio:
    # Read the stock from the csv
    df = pd.read_csv(f"HSI_1y.csv")
    # Generate portfolio with all given stocks and random proportion
    portfolio = Portfolio()
    # Add all given stocks with associated proportion
    for idx, tickers in enumerate(df['stock_id']):
        # try:
            portfolio.add_stock(Stock(tickers), solution[idx]/np.sum(solution))
        # except:
        #     print("error:\n", solution)
    return portfolio
# Using given weights, reduce the stock number
weight = [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]

num_generations = 100
num_stocks = 10
Asset_allocator(generated_portfolio(weight).filter_stock_by_sharpe_ratio(num_stocks), generations=num_generations).run()

def scatter_plot(history: list[float]):
    risk, roi = [], []
    # Get the portfolio from solution history
    for solution in history:
        port = generated_portfolio(weight).filter_stock_by_sharpe_ratio(num_stocks).set_weights(solution)
        # Get the sharpe ratio and risk for each 
        risk.append(port.risk)
        roi.append(port.roi)
    # print(risk, roi)

    # Create a gradient of colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(risk)))
    # Create the scatter plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.scatter(risk, roi, color=colors, marker='o')  # Create the scatter plot

    # Add labels and title
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.title('Efficient Frontier Approach to Optimal Portfolio')

    # Show the plot
    plt.savefig("results/Efficient_frontier.png", dpi=300)
    # plt.show()

model_hist = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").solution_history
model = Asset_allocator(Portfolio()).load_asset_allocator("storage/saved_asset_allocator").portfolio
print(model)
scatter_plot(model_hist)