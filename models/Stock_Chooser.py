import random
import pandas as pd
import numpy as np
from models import Stock, Portfolio

class Stock_chooser:
    def __init__(self, target_stocks, num_stocks, stocks_file, index="hsi", mutation_rate=0.1):
        self.target_stocks = target_stocks
        self.df = pd.read_csv(stocks_file)
        self.num_stocks = num_stocks
        self.preselected_stocks = self.preselect_stocks(index)
        self.mutation_rate = mutation_rate

    def preselect_stocks(self, index):
        # Calculate Sharpe ratio for each stock
        stocks = [Stock(ticker, index) for ticker in self.df['stock_id']]
        stock_sharpe_ratios = [(stock, stock.sharpe_ratio) for stock in stocks]
        
        # Sort stocks by Sharpe ratio in descending order
        sorted_stocks = sorted(stock_sharpe_ratios, key=lambda x: x[1], reverse=True)
        
        # Select top stocks based on num_stocks
        return [stock for stock, _ in sorted_stocks[:self.num_stocks]]

    def generated_portfolio(self, solution) -> Portfolio:
        portfolio = Portfolio()
        selected_stocks = [stock for stock, select in zip(self.preselected_stocks, solution) if select]
        for stock in selected_stocks:
            portfolio.add_stock(stock, 1/len(selected_stocks))
        return portfolio

    def single_point_crossover(self, a, b):
        if len(a) < 2:
            return a, b
        p = random.randint(1, len(a) - 1)
        return a[0:p] + b[p:], b[0:p] + a[p:]

    def select_pair(self, population, weights):
        return random.choices(
            population=population,
            weights=weights,
            k = 2
        )

    def mutation_func(self, genome):
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] = 1 - genome[i]  # Flip the bit
        
        # Ensure the correct number of stocks
        while sum(genome) != self.target_stocks:
            if sum(genome) > self.target_stocks:
                idx_ones = [i for i, gene in enumerate(genome) if gene == 1]
                idx = random.choice(idx_ones)
                genome[idx] = 0
            else:
                idx_zeros = [i for i, gene in enumerate(genome) if gene == 0]
                idx = random.choice(idx_zeros)
                genome[idx] = 1
        return genome

    def mating(self, parenta, parentb):
        offspring_a, offspring_b = self.single_point_crossover(parenta, parentb)
        offspring_a = self.mutation_func(offspring_a)
        offspring_b = self.mutation_func(offspring_b)
        return offspring_a, offspring_b

    def plot_fitness(self):
        # function to plot the fitness over generations
        return

    def plot_efficient_frontier(self):
        # function to plot the efficient frontier
        return

    def generate_initial_solution(self):
        solution = [0] * self.num_stocks
        ones = random.sample(range(self.num_stocks), self.target_stocks)
        for idx in ones:
            solution[idx] = 1
        return solution

    def genetic_algorithm(
            self,
            num_generations: int,
            sol_per_generation: int,
            keep_parents: int,
            print_generation: bool = True
    ):
        solutions = []
        generations = num_generations
        #  Generate the first population
        for _ in range(sol_per_generation):
            solutions.append(self.generate_initial_solution())

        for i in range(generations):
            rankedsolutions = []
            for s in solutions:
                #  Rank the existing solutions using the fitness function
                rankedsolutions.append((self.generated_portfolio(s).sharpe_ratio, s))
            rankedsolutions.sort(reverse=True)
            if print_generation:
                print(f"== Gen {i} best solutions ==")
                print(rankedsolutions[0])
                print(sum(rankedsolutions[0][1]))

            # take the trait of best solutions
            bestsolutions = rankedsolutions[:100]
            newGen = [rs[1] for rs in rankedsolutions[:keep_parents]]

            while len(newGen) < sol_per_generation:
                # randomly mutate the elements in parents and append to newGen
                parents = self.select_pair(bestsolutions, [solution[0] + abs(min([s[0] for s in bestsolutions])) for solution in bestsolutions])
                offspring_a, offspring_b = self.mating(parents[0][1], parents[1][1])
                newGen += [offspring_a, offspring_b]
            solutions = newGen
        
        # Return the portfolio with selected stocks instead of just the weights
        best_solution = rankedsolutions[0][1]
        return self.generated_portfolio(best_solution)

# # Example usage:
# chooser = Stock_chooser(target_stocks=15, num_stocks=30, stocks_file="HSI_1y.csv", mutation_rate=0.3)
# result = chooser.genetic_algorithm(num_generations=10, sol_per_generation=30, keep_parents=2)