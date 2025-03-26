import numpy as np
import random
import math
from statistics import median, mean, stdev
from collections import namedtuple

Solution = namedtuple('Solution', ['chromosome', 'x_values', 'objective'])

POPULATION_SIZE = 100
P_C = 0.9
P_M = 0.001
P_M_REAL = 0.01
NUM_GENERATIONS = 100
NUM_RUNS = 30

X_MIN = -4.5
X_MAX = 4.5

def random_binary_string(length):
    return ''.join(random.choice('01') for _ in range(length))

def decode_binary_chromosome(chromosome, x_min, x_max):
    x1_bits = chromosome[:14]
    x2_bits = chromosome[14:]
    
    x1_int = int(x1_bits, 2)
    x2_int = int(x2_bits, 2)
    
    max_int = 2**14 - 1
    x1 = x_min + (x_max - x_min) * x1_int / max_int
    x2 = x_min + (x_max - x_min) * x2_int / max_int
    
    return [x1, x2]

def beale_function(x):
    x1, x2 = x
    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*(x2**2))**2
    term3 = (2.625 - x1 + x1*(x2**3))**2
    return term1 + term2 + term3

def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + a + np.exp(1)

def fitness_beale(x):
    f = beale_function(x)
    return 1 / (f + 1)

def fitness_ackley(x):
    f = ackley_function(x)
    return 1 / (f + 1)

def stochastic_universal_sampling(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)
    
    distance = total_fitness / num_parents
    
    start = random.uniform(0, distance)
    
    pointers = [start + i * distance for i in range(num_parents)]
    
    selected = []
    for pointer in pointers:
        i = 0
        fitness_sum = fitnesses[0]
        while fitness_sum < pointer:
            i += 1
            if i >= len(fitnesses):
                i = len(fitnesses) - 1
                break
            fitness_sum += fitnesses[i]
        selected.append(i)
    
    return selected

def two_point_crossover(parent1, parent2):
    length = len(parent1)
    
    points = sorted(random.sample(range(1, length), 2))
    
    offspring1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    offspring2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    
    return offspring1, offspring2

def intermediate_crossover(parent1, parent2, alpha=0.5):
    offspring1 = []
    offspring2 = []
    
    for i in range(len(parent1)):
        w1 = random.uniform(0, 1+alpha)
        w2 = random.uniform(0, 1+alpha)
        
        gene1 = w1 * parent1[i] + (1 - w1) * parent2[i]
        gene2 = w2 * parent2[i] + (1 - w2) * parent1[i]
        
        gene1 = max(X_MIN, min(X_MAX, gene1))
        gene2 = max(X_MIN, min(X_MAX, gene2))
        
        offspring1.append(gene1)
        offspring2.append(gene2)
    
    return offspring1, offspring2

def bitwise_mutation(chromosome, p_m):
    mutated = ""
    for bit in chromosome:
        if random.random() < p_m:
            mutated += '1' if bit == '0' else '0'
        else:
            mutated += bit
    return mutated

def uniform_mutation(chromosome, p_m, x_min, x_max):
    mutated = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random() < p_m:
            mutated[i] = random.uniform(x_min, x_max)
    return mutated

def get_statistics(results):
    sorted_results = sorted(results, key=lambda x: x.objective)
    
    best_solution = sorted_results[0]
    worst_solution = sorted_results[-1]
    
    n = len(sorted_results)
    if n % 2 == 0:
        median_solution = sorted_results[n // 2 - 1]
    else:
        median_solution = sorted_results[n // 2]
    
    objective_values = [sol.objective for sol in results]
    mean_obj = mean(objective_values)
    std_dev_obj = stdev(objective_values) if len(objective_values) > 1 else 0
    
    return {
        'best': best_solution,
        'worst': worst_solution,
        'median': median_solution,
        'mean': mean_obj,
        'std_dev': std_dev_obj
    }

def binary_ga_beale(pop_size=POPULATION_SIZE, p_c=P_C, p_m=P_M, num_generations=NUM_GENERATIONS):
    population = [random_binary_string(28) for _ in range(pop_size)]
    
    best_solution = None
    
    for generation in range(num_generations):
        fitnesses = []
        solutions = []
        
        for chromosome in population:
            x_values = decode_binary_chromosome(chromosome, X_MIN, X_MAX)
            obj_value = beale_function(x_values)
            fitness = fitness_beale(x_values)
            
            fitnesses.append(fitness)
            solutions.append(Solution(chromosome, x_values, obj_value))
        
        generation_best = min(solutions, key=lambda s: s.objective)
        if best_solution is None or generation_best.objective < best_solution.objective:
            best_solution = generation_best
        
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i] for i in selected_indices]
        
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                if random.random() < p_c:
                    child1, child2 = two_point_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i], parents[i+1]
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i])
        
        for i in range(len(offspring)):
            offspring[i] = bitwise_mutation(offspring[i], p_m)
        
        offspring_decoded = [decode_binary_chromosome(chrom, X_MIN, X_MAX) for chrom in offspring]
        offspring_fitness = [fitness_beale(x) for x in offspring_decoded]
        offspring_obj_values = [beale_function(x) for x in offspring_decoded]
        
        worst_idx = offspring_obj_values.index(max(offspring_obj_values))
        
        if best_solution is not None and best_solution.objective < offspring_obj_values[worst_idx]:
            offspring[worst_idx] = best_solution.chromosome
        
        population = offspring
    
    return best_solution

def real_ga_beale(pop_size=POPULATION_SIZE, p_c=P_C, p_m=P_M_REAL, num_generations=NUM_GENERATIONS):
    population = [[random.uniform(X_MIN, X_MAX), random.uniform(X_MIN, X_MAX)] for _ in range(pop_size)]
    
    best_solution = None
    
    for generation in range(num_generations):
        fitnesses = []
        solutions = []
        
        for chromosome in population:
            obj_value = beale_function(chromosome)
            fitness = fitness_beale(chromosome)
            
            fitnesses.append(fitness)
            solutions.append(Solution(chromosome.copy(), chromosome.copy(), obj_value))
        
        generation_best = min(solutions, key=lambda s: s.objective)
        if best_solution is None or generation_best.objective < best_solution.objective:
            best_solution = generation_best
        
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i].copy() for i in selected_indices]
        
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                if random.random() < p_c:
                    child1, child2 = intermediate_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i].copy(), parents[i+1].copy()
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i].copy())
        
        for i in range(len(offspring)):
            offspring[i] = uniform_mutation(offspring[i], p_m, X_MIN, X_MAX)
        
        offspring_fitness = [fitness_beale(x) for x in offspring]
        offspring_obj_values = [beale_function(x) for x in offspring]
        
        worst_idx = offspring_obj_values.index(max(offspring_obj_values))
        
        if best_solution is not None and best_solution.objective < offspring_obj_values[worst_idx]:
            offspring[worst_idx] = best_solution.chromosome.copy()
        
        population = [chrom.copy() for chrom in offspring]
    
    return best_solution

def binary_ga_ackley(pop_size=POPULATION_SIZE, p_c=P_C, p_m=P_M, num_generations=NUM_GENERATIONS):
    population = [random_binary_string(28) for _ in range(pop_size)]
    
    best_solution = None
    
    for generation in range(num_generations):
        fitnesses = []
        solutions = []
        
        for chromosome in population:
            x_values = decode_binary_chromosome(chromosome, X_MIN, X_MAX)
            obj_value = ackley_function(x_values)
            fitness = fitness_ackley(x_values)
            
            fitnesses.append(fitness)
            solutions.append(Solution(chromosome, x_values, obj_value))
        
        generation_best = min(solutions, key=lambda s: s.objective)
        if best_solution is None or generation_best.objective < best_solution.objective:
            best_solution = generation_best
        
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i] for i in selected_indices]
        
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                if random.random() < p_c:
                    child1, child2 = two_point_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i], parents[i+1]
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i])
        
        for i in range(len(offspring)):
            offspring[i] = bitwise_mutation(offspring[i], p_m)
        
        offspring_decoded = [decode_binary_chromosome(chrom, X_MIN, X_MAX) for chrom in offspring]
        offspring_fitness = [fitness_ackley(x) for x in offspring_decoded]
        offspring_obj_values = [ackley_function(x) for x in offspring_decoded]
        
        worst_idx = offspring_obj_values.index(max(offspring_obj_values))
        
        if best_solution is not None and best_solution.objective < offspring_obj_values[worst_idx]:
            offspring[worst_idx] = best_solution.chromosome
        
        population = offspring
    
    return best_solution

def real_ga_ackley(pop_size=POPULATION_SIZE, p_c=P_C, p_m=P_M_REAL, num_generations=NUM_GENERATIONS, num_vars=2):
    population = [[random.uniform(X_MIN, X_MAX) for _ in range(num_vars)] for _ in range(pop_size)]
    
    best_solution = None
    
    for generation in range(num_generations):
        fitnesses = []
        solutions = []
        
        for chromosome in population:
            obj_value = ackley_function(chromosome)
            fitness = fitness_ackley(chromosome)
            
            fitnesses.append(fitness)
            solutions.append(Solution(chromosome.copy(), chromosome.copy(), obj_value))
        
        generation_best = min(solutions, key=lambda s: s.objective)
        if best_solution is None or generation_best.objective < best_solution.objective:
            best_solution = generation_best
        
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i].copy() for i in selected_indices]
        
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                if random.random() < p_c:
                    child1, child2 = intermediate_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i].copy(), parents[i+1].copy()
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i].copy())
        
        for i in range(len(offspring)):
            offspring[i] = uniform_mutation(offspring[i], p_m, X_MIN, X_MAX)
        
        offspring_fitness = [fitness_ackley(x) for x in offspring]
        offspring_obj_values = [ackley_function(x) for x in offspring]
        
        worst_idx = offspring_obj_values.index(max(offspring_obj_values))
        
        if best_solution is not None and best_solution.objective < offspring_obj_values[worst_idx]:
            offspring[worst_idx] = best_solution.chromosome.copy()
        
        population = [chrom.copy() for chrom in offspring]
    
    return best_solution

def run_experiment(ga_function, num_runs=NUM_RUNS, **kwargs):
    results = []
    
    for run in range(num_runs):
        solution = ga_function(**kwargs)
        results.append(solution)
    
    return get_statistics(results)

def print_comparative_table(results):
    print("\n" + "="*80)
    print(f"{'COMPARATIVE RESULTS':^80}")
    print("="*80)
    
    headers = ["Algorithm", "Problem", "Best Obj", "Best Solution", 
               "Worst Obj", "Median Obj", "Mean Obj", "Std Dev"]
    
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<25} "
          f"{headers[4]:<10} {headers[5]:<10} {headers[6]:<10} {headers[7]:<10}")
    print("-"*110)
    
    for algorithm, problems in results.items():
        for problem, stats in problems.items():
            best_sol_str = f"x={[round(x, 4) for x in stats['best'].x_values]}"
            print(f"{algorithm:<15} {problem:<10} {stats['best'].objective:<10.6f} "
                  f"{best_sol_str:<25} {stats['worst'].objective:<10.6f} "
                  f"{stats['median'].objective:<10.6f} {stats['mean']:<10.6f} "
                  f"{stats['std_dev']:<10.6f}")
    
    print("="*110)

def run_all_experiments():
    results = {
        "Binary GA": {
            "Beale": run_experiment(binary_ga_beale),
            "Ackley": run_experiment(binary_ga_ackley)
        },
        "Real GA": {
            "Beale": run_experiment(real_ga_beale),
            "Ackley-2D": run_experiment(real_ga_ackley, num_vars=2),
            "Ackley-5D": run_experiment(real_ga_ackley, num_vars=5),
            "Ackley-10D": run_experiment(real_ga_ackley, num_vars=10),
            "Ackley-20D": run_experiment(real_ga_ackley, num_vars=20)
        }
    }
    
    print_comparative_table(results)
    
    return results

def read_tsp_instance(file_path=None):
    if file_path:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    else:
        print("Enter TSP instance (first line: number of cities):")
        num_cities = int(input())
        print(f"Enter GA parameters (p_c, p_m, population_size):")
        params_line = input()
        
        lines = [str(num_cities), params_line]
        
        print(f"Enter cost matrix ({num_cities-1} lines):")
        for i in range(num_cities-1):
            lines.append(input())
    
    num_cities = int(lines[0].strip())
    
    params = list(map(float, lines[1].strip().split()))
    p_c, p_m, pop_size = params if len(params) == 3 else (0.9, 0.01, 100)
    
    cost_matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
    
    line_index = 2
    for i in range(num_cities - 1):
        costs = list(map(int, lines[line_index].strip().split()))
        for j, cost in enumerate(costs):
            cost_matrix[i][i+1+j] = cost
            cost_matrix[i+1+j][i] = cost
        line_index += 1
    
    return num_cities, p_c, p_m, int(pop_size), cost_matrix

def initialize_tsp_population(pop_size, num_cities):
    population = []
    
    for _ in range(pop_size):
        perm = list(range(1, num_cities))
        random.shuffle(perm)
        perm = [0] + perm
        population.append(perm)
    
    return population

def calculate_route_cost(route, cost_matrix):
    total_cost = 0
    
    for i in range(len(route) - 1):
        total_cost += cost_matrix[route[i]][route[i+1]]
    
    total_cost += cost_matrix[route[-1]][route[0]]
    
    return total_cost

def order_crossover(parent1, parent2):
    size = len(parent1)
    
    start, end = sorted(random.sample(range(1, size), 2))
    
    child1 = [-1] * size
    child2 = [-1] * size
    
    child1[0] = 0
    child2[0] = 0
    
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    
    fill_index1 = end if end < size else 1
    fill_index2 = end if end < size else 1
    
    for i in range(1, size):
        if child1[fill_index1] == -1:
            city = parent2[i]
            if city not in child1:
                child1[fill_index1] = city
                fill_index1 = (fill_index1 + 1) % size
                if fill_index1 == 0:
                    fill_index1 = 1
        
        if child2[fill_index2] == -1:
            city = parent1[i]
            if city not in child2:
                child2[fill_index2] = city
                fill_index2 = (fill_index2 + 1) % size
                if fill_index2 == 0:
                    fill_index2 = 1
    
    return child1, child2

def swap_mutation(route, p_m):
    mutated = route.copy()
    
    if random.random() < p_m:
        i, j = random.sample(range(1, len(route)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    
    return mutated

def tsp_ga(num_cities, cost_matrix, pop_size=100, p_c=0.9, p_m=0.01, num_generations=100):
    population = initialize_tsp_population(pop_size, num_cities)
    
    best_solution = None
    best_cost = float('inf')
    
    for generation in range(num_generations):
        route_costs = [calculate_route_cost(route, cost_matrix) for route in population]
        fitnesses = [1 / cost for cost in route_costs]
        
        min_cost_idx = route_costs.index(min(route_costs))
        if route_costs[min_cost_idx] < best_cost:
            best_cost = route_costs[min_cost_idx]
            best_solution = population[min_cost_idx].copy()
        
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i].copy() for i in selected_indices]
        
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                if random.random() < p_c:
                    child1, child2 = order_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i].copy(), parents[i+1].copy()
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i].copy())
        
        for i in range(len(offspring)):
            offspring[i] = swap_mutation(offspring[i], p_m)
        
        if best_solution:
            offspring_costs = [calculate_route_cost(route, cost_matrix) for route in offspring]
            
            worst_idx = offspring_costs.index(max(offspring_costs))
            
            if best_cost < offspring_costs[worst_idx]:
                offspring[worst_idx] = best_solution.copy()
        
        population = [route.copy() for route in offspring]
    
    return best_solution, best_cost

def run_tsp_experiment(num_cities, cost_matrix, num_runs=30, **kwargs):
    results = []
    
    for run in range(num_runs):
        route, cost = tsp_ga(num_cities, cost_matrix, **kwargs)
        results.append((route, cost))
    
    sorted_results = sorted(results, key=lambda x: x[1])
    
    costs = [result[1] for result in results]
    mean_cost = mean(costs)
    std_dev = stdev(costs) if len(costs) > 1 else 0
    
    best = sorted_results[0]
    worst = sorted_results[-1]
    
    n = len(sorted_results)
    median = sorted_results[n // 2] if n % 2 != 0 else sorted_results[n // 2 - 1]
    
    return {
        'best': best,
        'worst': worst,
        'median': median,
        'mean': mean_cost,
        'std_dev': std_dev
    }

def print_tsp_results(results):
    print("\n" + "="*80)
    print(f"{'TSP RESULTS':^80}")
    print("="*80)
    
    print(f"Best route: {results['best'][0]}")
    print(f"Best cost: {results['best'][1]}")
    print(f"Worst cost: {results['worst'][1]}")
    print(f"Median cost: {results['median'][1]}")
    print(f"Mean cost: {results['mean']:.2f}")
    print(f"Standard deviation: {results['std_dev']:.2f}")
    print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Genetic Algorithm Implementations")
    parser.add_argument('--function', type=str, choices=['beale', 'ackley', 'tsp', 'all'], 
                       default='all', help='Function to optimize')
    parser.add_argument('--tsp-file', type=str, help='TSP instance file')
    parser.add_argument('--runs', type=int, default=30, help='Number of runs')
    
    args = parser.parse_args()
    
    if args.function in ['beale', 'ackley', 'all']:
        run_all_experiments()
    
    if args.function in ['tsp', 'all']:
        if args.tsp_file:
            num_cities, p_c, p_m, pop_size, cost_matrix = read_tsp_instance(args.tsp_file)
        else:
            num_cities, p_c, p_m, pop_size, cost_matrix = read_tsp_instance()
        
        tsp_results = run_tsp_experiment(
            num_cities, cost_matrix, num_runs=args.runs,
            pop_size=pop_size, p_c=p_c, p_m=p_m
        )
        
        print_tsp_results(tsp_results)
        
        best_route = ''.join(map(str, tsp_results['best'][0]))
        best_cost = tsp_results['best'][1]
        print(f"\n{best_route}\n{best_cost}")

if __name__ == "__main__":
    main()