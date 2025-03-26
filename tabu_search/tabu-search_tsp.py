import random
import math
import statistics
import sys
import argparse
from typing import List, Tuple, Dict, Set, Optional

def read_input():
    N = int(input())
    Imax = int(input())
    
    cost_matrix = [[0 for _ in range(N)] for _ in range(N)]
    
    for i in range(N-1):
        values = list(map(int, input().split()))
        
        for j in range(len(values)):
            cost_matrix[i][i+j+1] = values[j]
            cost_matrix[i+j+1][i] = values[j]
    
    return N, Imax, cost_matrix

def read_instance_from_file(filename: str):
    with open(filename, 'r') as file:
        lines = file.readlines()
        N = int(lines[0].strip())
        Imax = int(lines[1].strip())
        
        cost_matrix = [[0 for _ in range(N)] for _ in range(N)]
        
        for i in range(N-1):
            values = list(map(int, lines[i+2].strip().split()))
            
            for j in range(len(values)):
                cost_matrix[i][i+j+1] = values[j]
                cost_matrix[i+j+1][i] = values[j]
    
    return N, Imax, cost_matrix

def greedy_initial_solution(N: int, cost_matrix: List[List[int]]) -> List[int]:
    route = [0]
    unvisited = list(range(1, N))
    
    while unvisited:
        current_city = route[-1]
        next_city = min(unvisited, key=lambda city: cost_matrix[current_city][city])
        route.append(next_city)
        unvisited.remove(next_city)
    
    return route

def calculate_cost(route: List[int], cost_matrix: List[List[int]]) -> int:
    total_cost = 0
    N = len(route)
    
    for i in range(N - 1):
        total_cost += cost_matrix[route[i]][route[i+1]]
    
    total_cost += cost_matrix[route[N-1]][route[0]]
    
    return total_cost

def tabu_search(N: int, Imax: int, cost_matrix: List[List[int]]) -> Tuple[List[int], int]:
    current_solution = greedy_initial_solution(N, cost_matrix)
    best_solution = current_solution.copy()
    current_cost = calculate_cost(current_solution, cost_matrix)
    best_cost = current_cost
    
    tabu_tenure = math.ceil(N / 2)
    tabu_list = {}
    
    for iteration in range(Imax):
        move_pos = random.randrange(1, N)
        move_city = current_solution[move_pos]
        
        best_neighbor = None
        best_neighbor_cost = float('inf')
        
        for new_pos in range(1, N):
            if new_pos == move_pos:
                continue
            
            neighbor = current_solution.copy()
            neighbor.pop(move_pos)
            neighbor.insert(new_pos, move_city)
            
            neighbor_cost = calculate_cost(neighbor, cost_matrix)
            
            move_key = (move_city, new_pos)
            is_tabu = move_key in tabu_list and tabu_list[move_key] > 0
            
            if is_tabu and neighbor_cost >= best_cost:
                continue
            
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
        
        if best_neighbor:
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
            
            move_city_idx = best_neighbor.index(move_city)
            tabu_list[(move_city, move_city_idx)] = tabu_tenure
            tabu_list[(move_city, move_pos)] = tabu_tenure
        
        expired_moves = []
        for move in tabu_list:
            tabu_list[move] -= 1
            if tabu_list[move] <= 0:
                expired_moves.append(move)
        
        for move in expired_moves:
            del tabu_list[move]
    
    return best_solution, best_cost

def multi_run_tabu_search(M: int, N: int, Imax: int, cost_matrix: List[List[int]]) -> tuple:
    results = []
    
    for run in range(M):
        print(f"Running iteration {run+1}/{M}...")
        best_route, best_cost = tabu_search(N, Imax, cost_matrix)
        results.append((best_route, best_cost))
    
    results.sort(key=lambda x: x[1])
    
    costs = [cost for _, cost in results]
    mean_cost = statistics.mean(costs)
    std_dev = statistics.stdev(costs) if len(costs) > 1 else 0
    
    best_solution = results[0]
    worst_solution = results[-1]
    
    if M % 2 == 0:
        median_idx = M // 2 - 1
    else:
        median_idx = M // 2
    
    median_solution = results[median_idx]
    
    return best_solution, worst_solution, median_solution, mean_cost, std_dev

def tabu_search_enhanced(N: int, Imax: int, cost_matrix: List[List[int]], 
                        dynamic_tenure: bool = False, 
                        diversification: bool = False, 
                        swap_moves: bool = False) -> Tuple[List[int], int]:
    current_solution = greedy_initial_solution(N, cost_matrix)
    best_solution = current_solution.copy()
    current_cost = calculate_cost(current_solution, cost_matrix)
    best_cost = current_cost
    
    base_tabu_tenure = math.ceil(N / 2)
    tabu_list = {}
    
    frequency = {(i, j): 0 for i in range(N) for j in range(N) if i != j} if diversification else None
    
    no_improvement = 0
    
    for iteration in range(Imax):
        use_swap = swap_moves and random.random() < 0.5
        
        best_neighbor = None
        best_neighbor_cost = float('inf')
        
        if use_swap:
            pos1 = random.randrange(1, N)
            pos2 = random.randrange(1, N)
            while pos1 == pos2:
                pos2 = random.randrange(1, N)
            
            for _ in range(1):
                neighbor = current_solution.copy()
                neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
                
                neighbor_cost = calculate_cost(neighbor, cost_matrix)
                
                if diversification:
                    for i in range(N-1):
                        from_city = neighbor[i]
                        to_city = neighbor[i+1]
                        neighbor_cost += 0.01 * frequency[(from_city, to_city)]
                
                city1, city2 = current_solution[pos1], current_solution[pos2]
                swap_key1 = (city1, pos2)
                swap_key2 = (city2, pos1)
                is_tabu = (swap_key1 in tabu_list and tabu_list[swap_key1] > 0) or \
                          (swap_key2 in tabu_list and tabu_list[swap_key2] > 0)
                
                if is_tabu and neighbor_cost >= best_cost:
                    continue
                
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
        else:
            move_pos = random.randrange(1, N)
            move_city = current_solution[move_pos]
            
            for new_pos in range(1, N):
                if new_pos == move_pos:
                    continue
                
                neighbor = current_solution.copy()
                neighbor.pop(move_pos)
                neighbor.insert(new_pos, move_city)
                
                neighbor_cost = calculate_cost(neighbor, cost_matrix)
                
                if diversification:
                    for i in range(N-1):
                        from_city = neighbor[i]
                        to_city = neighbor[i+1]
                        neighbor_cost += 0.01 * frequency[(from_city, to_city)]
                
                move_key = (move_city, new_pos)
                is_tabu = move_key in tabu_list and tabu_list[move_key] > 0
                
                if is_tabu and neighbor_cost >= best_cost:
                    continue
                
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
        
        if best_neighbor:
            if diversification:
                for i in range(N-1):
                    from_city = best_neighbor[i]
                    to_city = best_neighbor[i+1]
                    frequency[(from_city, to_city)] += 1
                from_city = best_neighbor[-1]
                to_city = best_neighbor[0]
                frequency[(from_city, to_city)] += 1
            
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
                no_improvement = 0
            else:
                no_improvement += 1
            
            if dynamic_tenure:
                tenure = base_tabu_tenure + min(no_improvement // 5, base_tabu_tenure)
            else:
                tenure = base_tabu_tenure
            
            if use_swap:
                pos1, pos2 = -1, -1
                for i in range(N):
                    if current_solution[i] != best_neighbor[i]:
                        if pos1 == -1:
                            pos1 = i
                        else:
                            pos2 = i
                            break
                
                if pos1 != -1 and pos2 != -1:
                    city1, city2 = best_neighbor[pos1], best_neighbor[pos2]
                    tabu_list[(city1, pos2)] = tenure
                    tabu_list[(city2, pos1)] = tenure
            else:
                for i in range(N):
                    if current_solution[i] != best_neighbor[i]:
                        city = current_solution[i] if current_solution[i] not in best_neighbor[:i+1] else best_neighbor[i]
                        old_pos = current_solution.index(city)
                        new_pos = best_neighbor.index(city)
                        tabu_list[(city, old_pos)] = tenure
                        break
        
        expired_moves = []
        for move in tabu_list:
            tabu_list[move] -= 1
            if tabu_list[move] <= 0:
                expired_moves.append(move)
        
        for move in expired_moves:
            del tabu_list[move]
    
    return best_solution, best_cost

def main():
    parser = argparse.ArgumentParser(description='Tabu Search for TSP')
    parser.add_argument('--mode', type=str, choices=['basic', 'extension', 'challenge'], 
                        default='basic', help='Mode to run')
    parser.add_argument('--file', type=str, help='Instance file for extension/challenge modes')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for extension/challenge modes')
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        N, Imax, cost_matrix = read_input()
        best_route, best_cost = tabu_search(N, Imax, cost_matrix)
        
        route_str = ' '.join(map(str, best_route))
        print(f"{route_str} {best_cost}")
    
    elif args.mode == 'extension':
        if args.file:
            filename = args.file
        else:
            filename = input("Enter the filename of the instance: ")
        
        M = args.runs if args.runs else int(input("Enter the number of executions: "))
        
        N, Imax, cost_matrix = read_instance_from_file(filename)
        best, worst, median, mean, std_dev = multi_run_tabu_search(M, N, Imax, cost_matrix)
        
        print("\nResults:")
        print("Best solution:", ' '.join(map(str, best[0])), best[1])
        print("Worst solution:", ' '.join(map(str, worst[0])), worst[1])
        print("Median solution:", ' '.join(map(str, median[0])), median[1])
        print(f"Mean cost: {mean:.2f}")
        print(f"Standard deviation: {std_dev:.2f}")
    
    elif args.mode == 'challenge':
        if args.file:
            filename = args.file
        else:
            filename = input("Enter the filename of the instance: ")
        
        M = args.runs if args.runs else int(input("Enter the number of executions: "))
        
        N, Imax, cost_matrix = read_instance_from_file(filename)
        
        strategies = [
            ("Basic Tabu Search", False, False, False),
            ("With Dynamic Tenure", True, False, False),
            ("With Diversification", False, True, False),
            ("With Swap Moves", False, False, True),
            ("Full Enhancement", True, True, True)
        ]
        
        best_results = []
        
        for name, dynamic, diversification, swap in strategies:
            print(f"\nRunning {name}...")
            
            results = []
            for run in range(M):
                print(f"  Run {run+1}/{M}...")
                best_route, best_cost = tabu_search_enhanced(
                    N, Imax, cost_matrix, dynamic, diversification, swap
                )
                results.append((best_route, best_cost))
            
            costs = [cost for _, cost in results]
            mean_cost = statistics.mean(costs)
            std_dev = statistics.stdev(costs) if len(costs) > 1 else 0
            
            best_result = min(results, key=lambda x: x[1])
            best_results.append((name, best_result, mean_cost, std_dev))
            
            print(f"  Best cost: {best_result[1]}")
            print(f"  Mean cost: {mean_cost:.2f}")
            print(f"  Std Dev: {std_dev:.2f}")
        
        print("\nComparison of Strategies:")
        for name, (route, cost), mean, std_dev in best_results:
            print(f"{name}: Best={cost}, Mean={mean:.2f}, StdDev={std_dev:.2f}")

if __name__ == "__main__":
    main()