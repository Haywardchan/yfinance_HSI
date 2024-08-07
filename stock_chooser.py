# Genetic Algorithm build from scratch
import random
import pandas as pd
import numpy as np
from models import Stock, Portfolio

def generated_portfolio(solution) -> Portfolio:
    # Read the stock from the csv
    df = pd.read_csv(f"HSI_1y.csv")
    # Generate portfolio with all given stocks and random proportion
    portfolio = Portfolio()
    # Add all given stocks with associated proportion
    for idx, tickers in enumerate(df['stock_id']):
        try:
            portfolio.add_stock(Stock(tickers), solution[idx]/np.sum(solution))
        except:
            print("error:\n", solution)
    return portfolio

def single_point_crossover(a, b):
    if len(a) < 2:
        return a. b
    p = random.randint(1, len(a) - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def select_pair(population, weights):
    return random.choices(
        population=population,
        weights=weights,
        k = 2
    )

def mutation_func(genome, num = 1, probability = 0.5):
    for _ in range(num):
        idx = random.randrange(len(genome))
        genome[idx] = genome[idx] if random.random() > probability else 1-genome[idx]
    return genome

def mating(parenta, parentb, num_crossover = 1, num_mutation = 1):
    for _ in range(num_crossover):
        offspring_a, offspring_b = single_point_crossover(parenta, parentb)
        parenta, parentb = offspring_a, offspring_b
    for _ in range(num_mutation):
        offspring_a = mutation_func(offspring_a)
        offspring_b = mutation_func(offspring_b)
    return offspring_a, offspring_b

def plot_fitness():
    # function to plot the fitness over generations
    return

def plot_efficient_frontier():
    # function to plot the efficient frontier
    return

def genetic_algorithm(
        num_generations: int,
        num_genes: int,
        sol_per_generation: int,
        keep_parents: int,
        print_generation: bool = True
):
    solutions = []
    generations = num_generations
    num_stock = num_genes
    #  Generate the first population
    for i in range(sol_per_generation):
        solutions.append([random.choice([0, 1]) for _ in range(num_stock)])

    for i in range(generations):
        rankedsolutions = []
        for s in solutions:
            #  Rank the existing solutions using the fitness function
            rankedsolutions.append((generated_portfolio(s).sharpe_ratio, s)) # Can substitute generated_portfolio(s).sharpe_ratio as mutation function
        rankedsolutions.sort()
        rankedsolutions.reverse()
        if(print_generation):
            print(f"== Gen {i} best solutions ==")
            print(rankedsolutions[0])
            print(np.sum(rankedsolutions[0][1]))

        # take the trait of best solutions
        bestsolutions = rankedsolutions[:100]
        newGen = [rs[1] for rs in rankedsolutions[:keep_parents]]

        for _ in range(int(len(bestsolutions)/2) - 1):
            # randomly mutate the elements in parents and append to newGen
            parents = select_pair(bestsolutions, [solution[0] + abs(min([s[0] for s in bestsolutions])) for solution in bestsolutions])
            offspring_a, offspring_b = mating(parents[0][1], parents[1][1])
            newGen += [offspring_a, offspring_b]
        solutions = newGen
    return rankedsolutions[0]

genetic_algorithm(num_generations=100, num_genes=pd.read_csv(f"HSI_1y.csv").shape[0], sol_per_generation=10, keep_parents=2)