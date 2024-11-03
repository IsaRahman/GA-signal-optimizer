import numpy as np
import random
from regression_models import (GA_Through_regression, GA_Delay_regression,
                               MA_Through_regression, MA_Delay_regression,
                               GNA_Through_regression, GNA_Delay_regression,
                               KA_Through_regression, KA_Delay_regression,
                               GA_Through_degree, GA_Delay_degree,
                               MA_Through_degree, MA_Delay_degree,
                               GNA_Through_degree, GNA_Delay_degree,
                               KA_Through_degree, KA_Delay_degree)
from sklearn.preprocessing import PolynomialFeatures






# Constants
cycle_length_min = 140
cycle_length_max = 200


num_links = 4  # Number of links or phases

# GA Parameters
population_size = 100  # Adjust as needed
crossover_rate = 0.8  # Adjust as needed
mutation_rate = 0.1  # Adjust as needed


# Fitness function
def fitness_function(chromosome, cycle_length):
    total_throughput = 0
    total_delay_time = 0
    short_green_time_penalty = 0
    
    GA_Green_time = chromosome[0]  # X1
    MA_Green_time = chromosome[1]  # X2
    GNA_Green_time = chromosome[2]  # X3
    KA_Green_time = chromosome[3]   # X4
    
    # Calculate throughput
    total_throughput += calculate_throughput(GA_Green_time, GA_Through_degree, GA_Through_regression)
    total_throughput += calculate_throughput(MA_Green_time, MA_Through_degree, MA_Through_regression)
    total_throughput += calculate_throughput(GNA_Green_time, GNA_Through_degree, GNA_Through_regression)
    total_throughput += calculate_throughput(KA_Green_time, KA_Through_degree, KA_Through_regression)
    
    # Calculate delay
    total_delay_time += calculate_delay(GA_Green_time, GA_Delay_degree, GA_Delay_regression, cycle_length)
    total_delay_time += calculate_delay(MA_Green_time, MA_Delay_degree, MA_Delay_regression, cycle_length)
    total_delay_time += calculate_delay(GNA_Green_time, GNA_Delay_degree, GNA_Delay_regression, cycle_length)
    total_delay_time += calculate_delay(KA_Green_time, KA_Delay_degree, KA_Delay_regression, cycle_length)
    
    # # Short green time penalty
    # for green_time in [GA_Green_time, MA_Green_time, GNA_Green_time, KA_Green_time]:
    #     if green_time < 15:
    #         short_green_time_penalty += (15 - green_time) * 10
    
    # Calculate average delay time
    average_delay_time = total_delay_time / cycle_length
    Avg_through= (total_throughput*cycle_length)/3600
    
    # Calculate fitness score
    fitness_score = Avg_through - 0.5 * average_delay_time
    
    return fitness_score


def calculate_throughput(green_time, degree, model):
    poly_features = PolynomialFeatures(degree)
    array = np.array(green_time)
    poly_values = poly_features.fit_transform(array.reshape(-1, 1))
    throughput = model.predict(poly_values)[0]
    return throughput

def calculate_delay(green_time, degree, model, cycle_length):
    poly_features = PolynomialFeatures(degree)
    array = np.array(green_time)
    poly_values = poly_features.fit_transform(array.reshape(-1, 1))
    delay = model.predict(poly_values)[0] * (cycle_length - green_time)
    return delay

class Chromosome:
    def __init__(self, genes, cycle_length):
        self.genes = genes
        self.cycle_length = cycle_length
        self.balance_genes()
        self.fitness = fitness_function(genes, cycle_length)

    def copy(self):
        return Chromosome(self.genes.copy(), self.cycle_length)

    def balance_genes(self):
        total_time = sum(self.genes)
        if total_time > self.cycle_length:
            excess = total_time - self.cycle_length
            while excess > 0:
                for i in range(len(self.genes)):
                    if self.genes[i] > 10:
                        reduction = min(excess, self.genes[i] - 10)
                        self.genes[i] -= reduction
                        excess -= reduction
                        if excess <= 0:
                            break
        elif total_time < self.cycle_length:
            deficit = self.cycle_length - total_time
            while deficit > 0:
                for i in range(len(self.genes)):
                    addition = min(deficit, 80 - self.genes[i])
                    self.genes[i] += addition
                    deficit -= addition
                    if deficit <= 0:
                        break



def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1.genes) - 2)
        child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child = Chromosome(child_genes, parent1.cycle_length)
        child.balance_genes()
        return child
    else:
        return parent1.copy()


def selection(population):
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_population[:population_size]

def mutation(chromosome):
    if random.random() < mutation_rate:
        mutation_point = random.randint(0, len(chromosome.genes) - 1)
        original_time = chromosome.genes[mutation_point]
        max_possible = min(80, chromosome.cycle_length - sum(chromosome.genes) + original_time)
        min_possible = max(10, chromosome.cycle_length - sum(chromosome.genes) + original_time - 80 * (len(chromosome.genes) - 1))
        if min_possible <= max_possible:
            new_green_time = random.randint(min_possible, max_possible)
            chromosome.genes[mutation_point] = new_green_time
            chromosome.balance_genes()



def generate_initial_green_times(cycle_length):
    remaining_time = cycle_length
    green_times = []
    
    for i in range(num_links):
        if remaining_time <= 0:
            green_times.append(0)
            continue
        if i == num_links - 1:
            # Assign all remaining time to the last phase ensuring it is within bounds
            green_time = max(min(remaining_time, 80), 10)
        else:
            max_possible = min(80, remaining_time - 10 * (num_links - i - 1))  # Ensure space for minimums in later phases
            min_possible = min(10, remaining_time)
            green_time = random.randint(min_possible, max_possible)
        green_times.append(green_time)
        remaining_time -= green_time
    
    return green_times



# Main loop function
def main():
    for _ in range(1):
        cycle_length = random.randint(cycle_length_min, cycle_length_max)
        population = [Chromosome(generate_initial_green_times(cycle_length), cycle_length) for _ in range(population_size)]

        for generation in range(50):  # Number of generations (adjust as needed)
            new_population = selection(population)

            for i in range(0, len(new_population) - 1, 2):
                parent1 = new_population[i]
                parent2 = new_population[i + 1]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                new_population.append(child1)
                new_population.append(child2)

            for individual in new_population:
                mutation(individual)

            for individual in new_population:
                individual.fitness = fitness_function(individual.genes, individual.cycle_length)

            population = sorted(new_population, key=lambda x: x.fitness, reverse=True)[:population_size]

        # Results
        best_chromosome = population[0]
        total_green_time = sum(best_chromosome.genes)

        # print(f"Cycle Length: {cycle_length} seconds")
        print(f" {cycle_length} , {best_chromosome.fitness} ")
        print(f"Best Green Times (seconds): {best_chromosome.genes}")
        # print(f"Total Green Time Utilized: {total_green_time} seconds out of {cycle_length} seconds")
        # print(f"The Fitness score is: {best_chromosome.fitness} .")
      

if __name__ == "__main__":
    main()

