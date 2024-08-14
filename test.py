# Example usage:
from models.Stock_Chooser import Stock_chooser
from models.Asset_Allocator import Asset_allocator

chooser = Stock_chooser(target_stocks=15, num_stocks=30, stocks_file="HSI_1y.csv", mutation_rate=0.3)
portfolio = chooser.genetic_algorithm(num_generations=10, sol_per_generation=30, keep_parents=2)
model = Asset_allocator(portfolio, generations=10).run()