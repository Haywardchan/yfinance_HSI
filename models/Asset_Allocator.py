import random
from models import Stock, Portfolio
import pandas as pd
import numpy as np
import pygad
import pickle

class Asset_allocator:
    def __init__(self, portfolio: Portfolio, generations: int = 10):
        self.portfolio = portfolio
        self.epoch: int = 0
        self.num_generations = generations
        self.ga_instance = None
        self.solution_history: list[float] = [] 

    def load_ga_instance(self, filename) -> pygad.GA:
        self.ga_instance = pygad.load(filename)
        return self.ga_instance
    
    def save_asset_allocator(self, filename):
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self, f)
    
    def load_asset_allocator(self, filename):
        with open(filename + ".pkl", "rb") as f:
            return pickle.load(f)

    def generated_portfolio(self, solution) -> Portfolio:
        self.portfolio.set_weights(solution)
        return self.portfolio

    def fitness_func(self, ga_instance, solution, solution_idx):
        gene_value = solution / np.sum(solution)
        # Set constraint
        for gene in gene_value:
            if gene < 0 or gene > 2/len(self.portfolio.get_weights()):
                return -1
        return self.generated_portfolio(gene_value).sharpe_ratio

    def on_generation(self, ga_instance):
        """
        Callback function called after each generation.
        Stores the best solution for the current generation.
        """
        # Get the best solution for the current generation
        best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
        self.solution_history += [np.array(best_solution / np.sum(best_solution))] # append best solution portfolio in each epoch
        # Print the results
        print(f"== Generation {self.epoch} ==")
        print(f"Sharpe Ratio: {best_fitness}")
        print(f"Solution: {np.array(best_solution / np.sum(best_solution))}")
        self.epoch += 1
        # print(f"solution_history {self.portfolio.set_weights(np.array(best_solution / np.sum(best_solution)))}\n")

    def run(self):
        ga_instance = pygad.GA(
            num_generations = self.num_generations,
            fitness_func = self.fitness_func,
            num_parents_mating = 20,
            sol_per_pop = 30,
            num_genes = len(self.portfolio.get_weights()),
            keep_parents = 20,
            on_generation = self.on_generation,
            init_range_low = 0,
            init_range_high = 1,
            random_mutation_min_val = -0.2,
            random_mutation_max_val = 1,
            # gene_space = {'low': 0, 'high': 2/len(self.portfolio.get_weights())}
        )
        ga_instance.run()
        ga_instance.plot_fitness(label=['Fitness', 'Sharpe_Ratio'])
        best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()
        self.portfolio.set_weights(self.solution_history[-1])
        self.ga_instance = ga_instance
        ga_instance.save("storage/saved_ga_instance")
        self.save_asset_allocator("storage/saved_asset_allocator")
        return self.solution_history[-1]
