import random
import math
import numpy as np
from typing import List, Tuple, Dict, Any
import os


class KnapsackSA:
    def __init__(self, initial_temp: float, final_temp: float, capacity: int, objects: List[Tuple[int, int]]):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.capacity = capacity
        self.objects = objects
        self.n = len(objects)
        self.cooling_rate = 0.99
        
    def calculate_weight(self, solution: List[int]) -> int:
        total_weight = 0
        for i in range(self.n):
            if solution[i] == 1:
                total_weight += self.objects[i][1]
        return total_weight
    
    def calculate_value(self, solution: List[int]) -> int:
        total_value = 0
        for i in range(self.n):
            if solution[i] == 1:
                total_value += self.objects[i][0]
        return total_value
    
    def is_feasible(self, solution: List[int]) -> bool:
        return self.calculate_weight(solution) <= self.capacity
    
    def generate_initial_solution(self) -> List[int]:
        solution = [0] * self.n
        indices = list(range(self.n))
        random.shuffle(indices)
        
        for i in indices:
            solution[i] = 1
            if not self.is_feasible(solution):
                solution[i] = 0
                
        return solution
    
    def generate_neighborhood(self, solution: List[int]) -> List[List[int]]:
        neighborhood = []
        
        for i in range(self.n):
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]
            if self.is_feasible(neighbor):
                neighborhood.append(neighbor)
                
        return neighborhood
    
    def simulated_annealing(self) -> Dict[str, Any]:
        current_solution = self.generate_initial_solution()
        best_solution = current_solution.copy()
        current_value = self.calculate_value(current_solution)
        best_value = current_value
        
        current_temp = self.initial_temp
        
        while current_temp >= self.final_temp:
            neighborhood = self.generate_neighborhood(current_solution)
            
            if not neighborhood:
                break
                
            neighbor = random.choice(neighborhood)
            neighbor_value = self.calculate_value(neighbor)
            
            delta = neighbor_value - current_value
            
            if delta > 0 or random.random() < math.exp(delta / current_temp):
                current_solution = neighbor
                current_value = neighbor_value
                
                if current_value > best_value:
                    best_solution = current_solution.copy()
                    best_value = current_value
            
            current_temp *= self.cooling_rate
        
        indices = [i for i in range(self.n) if best_solution[i] == 1]
        
        best_weight = self.calculate_weight(best_solution)
        
        return {
            "indices": indices,
            "solution": best_solution,
            "value": best_value,
            "weight": best_weight
        }


class ImprovedKnapsackSA(KnapsackSA):
    def __init__(self, initial_temp: float, final_temp: float, capacity: int, objects: List[Tuple[int, int]]):
        super().__init__(initial_temp, final_temp, capacity, objects)
        self.base_cooling_rate = 0.99
        
        self.value_weight_ratios = [(i, objects[i][0] / max(1, objects[i][1])) for i in range(self.n)]
        self.value_weight_ratios.sort(key=lambda x: x[1], reverse=True)
        
    def generate_greedy_initial_solution(self) -> List[int]:
        solution = [0] * self.n
        current_weight = 0
        
        for idx, _ in self.value_weight_ratios:
            if current_weight + self.objects[idx][1] <= self.capacity:
                solution[idx] = 1
                current_weight += self.objects[idx][1]
                
        return solution
    
    def generate_enhanced_neighborhood(self, solution: List[int], temp_ratio: float) -> List[List[int]]:
        neighborhood = []
        
        for i in range(self.n):
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]
            if self.is_feasible(neighbor):
                neighborhood.append(neighbor)
        
        if temp_ratio < 0.7:
            sample_size = min(self.n // 2, 10)
            first_indices = random.sample(range(self.n), sample_size)
            
            for i in first_indices:
                for j in range(i+1, min(i+5, self.n)):
                    neighbor = solution.copy()
                    neighbor[i] = 1 - neighbor[i]
                    neighbor[j] = 1 - neighbor[j]
                    if self.is_feasible(neighbor):
                        neighborhood.append(neighbor)
        
        if temp_ratio < 0.3:
            current_weight = self.calculate_weight(solution)
            
            not_included = [(i, self.objects[i][0]) for i in range(self.n) if solution[i] == 0]
            not_included.sort(key=lambda x: x[1], reverse=True)
            
            included = [(i, self.objects[i][0]) for i in range(self.n) if solution[i] == 1]
            included.sort(key=lambda x: x[1])
            
            for i_idx, i_val in not_included[:5]:
                for j_idx, j_val in included[:5]:
                    if i_val > j_val:
                        weight_diff = self.objects[i_idx][1] - self.objects[j_idx][1]
                        if current_weight + weight_diff <= self.capacity:
                            neighbor = solution.copy()
                            neighbor[j_idx] = 0
                            neighbor[i_idx] = 1
                            neighborhood.append(neighbor)
        
        return neighborhood
    
    def adaptive_cooling_rate(self, iteration: int, no_improvement_count: int, temp_ratio: float) -> float:
        cooling_rate = self.base_cooling_rate
        
        if iteration < 100:
            cooling_rate = min(0.995, cooling_rate)
            
        if no_improvement_count > 50:
            cooling_rate = max(0.95, cooling_rate - 0.01)
            
        if temp_ratio < 0.1:
            cooling_rate = min(0.999, cooling_rate + 0.005)
            
        return cooling_rate
    
    def simulated_annealing(self) -> Dict[str, Any]:
        current_solution = self.generate_greedy_initial_solution()
        best_solution = current_solution.copy()
        current_value = self.calculate_value(current_solution)
        best_value = current_value
        
        current_temp = self.initial_temp
        iteration = 0
        no_improvement_count = 0
        
        while current_temp >= self.final_temp:
            iteration += 1
            temp_ratio = current_temp / self.initial_temp
            
            neighborhood = self.generate_enhanced_neighborhood(current_solution, temp_ratio)
            
            if not neighborhood:
                break
                
            neighbor = random.choice(neighborhood)
            neighbor_value = self.calculate_value(neighbor)
            
            delta = neighbor_value - current_value
            
            if delta > 0 or random.random() < math.exp(delta / (current_temp * (1 + 0.1 * temp_ratio))):
                current_solution = neighbor
                current_value = neighbor_value
                
                if current_value > best_value:
                    best_solution = current_solution.copy()
                    best_value = current_value
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            if no_improvement_count > 100 and temp_ratio < 0.5:
                current_temp = min(self.initial_temp * 0.5, current_temp * 2)
                no_improvement_count = 0
            else:
                cooling_rate = self.adaptive_cooling_rate(iteration, no_improvement_count, temp_ratio)
                current_temp *= cooling_rate
        
        indices = [i for i in range(self.n) if best_solution[i] == 1]
        
        best_weight = self.calculate_weight(best_solution)
        
        return {
            "indices": indices,
            "solution": best_solution,
            "value": best_value,
            "weight": best_weight
        }


def parse_input(input_data: str) -> Tuple[float, float, int, List[Tuple[int, int]]]:
    lines = input_data.strip().split('\n')
    
    initial_temp, final_temp = map(float, lines[0].split())
    
    n = int(lines[1])
    
    capacity = int(lines[2])
    
    objects = []
    for i in range(3, 3 + n):
        value, weight = map(int, lines[i].split())
        objects.append((value, weight))
    
    return initial_temp, final_temp, capacity, objects


def run_single_execution(input_data: str, use_improved: bool = False) -> Dict[str, Any]:
    initial_temp, final_temp, capacity, objects = parse_input(input_data)
    
    if use_improved:
        solver = ImprovedKnapsackSA(initial_temp, final_temp, capacity, objects)
    else:
        solver = KnapsackSA(initial_temp, final_temp, capacity, objects)
    
    result = solver.simulated_annealing()
    
    print(f"Indices of objects included: {result['indices']}")
    print(f"Binary solution: {''.join(map(str, result['solution']))}")
    print(f"Total value: {result['value']}")
    print(f"Total weight: {result['weight']}")
    
    return result


def run_multiple_executions(input_data: str, m: int, use_improved: bool = False) -> Dict:
    initial_temp, final_temp, capacity, objects = parse_input(input_data)
    
    results = []
    
    for i in range(1, m + 1):
        print(f"Execution {i}/{m}...")
        if use_improved:
            solver = ImprovedKnapsackSA(initial_temp, final_temp, capacity, objects)
        else:
            solver = KnapsackSA(initial_temp, final_temp, capacity, objects)
            
        result = solver.simulated_annealing()
        results.append(result)
    
    obj_values = [result["value"] for result in results]
    
    sorted_results = sorted(results, key=lambda x: x["value"])
    
    best_result = sorted_results[-1]
    worst_result = sorted_results[0]
    median_result = sorted_results[m // 2]
    mean_value = np.mean(obj_values)
    std_dev = np.std(obj_values)
    
    print("\n=== Results of", m, "executions ===")
    
    print("\nBest Solution:")
    print(f"Indices of objects included: {best_result['indices']}")
    print(f"Binary solution: {''.join(map(str, best_result['solution']))}")
    print(f"Total value: {best_result['value']}")
    print(f"Total weight: {best_result['weight']}")
    
    print("\nWorst Solution:")
    print(f"Indices of objects included: {worst_result['indices']}")
    print(f"Binary solution: {''.join(map(str, worst_result['solution']))}")
    print(f"Total value: {worst_result['value']}")
    print(f"Total weight: {worst_result['weight']}")
    
    print("\nMedian Solution:")
    print(f"Indices of objects included: {median_result['indices']}")
    print(f"Binary solution: {''.join(map(str, median_result['solution']))}")
    print(f"Total value: {median_result['value']}")
    print(f"Total weight: {median_result['weight']}")
    
    print("\nStatistical Metrics:")
    print(f"Mean Objective Value: {mean_value}")
    print(f"Standard Deviation: {std_dev}")
    
    return {
        "best": best_result,
        "worst": worst_result,
        "median": median_result,
        "mean": mean_value,
        "std_dev": std_dev
    }


def compare_algorithms(input_data: str, m: int) -> Dict:
    print("=== Comparing Basic and Improved SA Algorithms ===")
    
    print("\nRunning Basic SA Algorithm...")
    basic_stats = run_multiple_executions(input_data, m, use_improved=False)
    
    print("\nRunning Improved SA Algorithm...")
    improved_stats = run_multiple_executions(input_data, m, use_improved=True)
    
    mean_improvement = (improved_stats["mean"] - basic_stats["mean"]) / basic_stats["mean"] * 100
    max_improvement = (improved_stats["best"]["value"] - basic_stats["best"]["value"]) / basic_stats["best"]["value"] * 100
    
    print("\n=== Improvement Summary ===")
    print(f"Mean Value Improvement: {mean_improvement:.2f}%")
    print(f"Best Value Improvement: {max_improvement:.2f}%")
    
    return {
        "basic": basic_stats,
        "improved": improved_stats,
        "mean_improvement": mean_improvement,
        "max_improvement": max_improvement
    }


def main():
    print("=== Simulated Annealing for the Knapsack Problem ===")
    
    mode = input("Select mode (1 for single execution, 2 for multiple executions, 3 for algorithm comparison): ")
    
    if mode == '1':
        input_method = input("Input method (1 for manual input, 2 for file input): ")
        use_improved = input("Use improved algorithm? (y/n): ").lower() == 'y'
        
        if input_method == '1':
            print("Enter input parameters:")
            init_temp = float(input("Initial temperature: "))
            final_temp = float(input("Final temperature: "))
            n = int(input("Number of objects: "))
            capacity = int(input("Knapsack capacity: "))
            
            objects = []
            print("Enter value and weight for each object:")
            for i in range(n):
                value, weight = map(int, input(f"Object {i+1} (value weight): ").split())
                objects.append((value, weight))
            
            input_data = f"{init_temp} {final_temp}\n{n}\n{capacity}\n"
            for value, weight in objects:
                input_data += f"{value} {weight}\n"
                
            run_single_execution(input_data, use_improved)
            
        elif input_method == '2':
            filename = input("Enter the input file name: ")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    input_data = f.read()
                run_single_execution(input_data, use_improved)
            except FileNotFoundError:
                print(f"Error: Could not find file '{filename}' in {script_dir}")
                return
            
        else:
            print("Invalid input method.")
            
    elif mode == '2':
        filename = input("Enter the input file name: ")
        m = int(input("Enter the number of executions (M): "))
        use_improved = input("Use improved algorithm? (y/n): ").lower() == 'y'
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                input_data = f.read()
            run_multiple_executions(input_data, m, use_improved)
        except FileNotFoundError:
            print(f"Error: Could not find file '{filename}' in {script_dir}")
            return
        
    elif mode == '3':
        filename = input("Enter the input file name: ")
        m = int(input("Enter the number of executions for each algorithm (M): "))
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                input_data = f.read()
            compare_algorithms(input_data, m)
        except FileNotFoundError:
            print(f"Error: Could not find file '{filename}' in {script_dir}")
            return
        
    else:
        print("Invalid mode.")

if __name__ == "__main__":
    main()