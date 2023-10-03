# code the clique problem graph coloring use the genetic algorithm

import copy
import random

import matplotlib.pyplot as plt
import numpy as np


# define the class of the graph
class Graph:
    def __init__(self, matrix, color):
        self.matrix = matrix
        self.color = color

    def get_matrix(self):
        return self.matrix

    def get_color(self):
        return self.color

    def set_matrix(self, matrix):
        self.matrix = matrix

    def set_color(self, color):
        self.color = color


# define the class of the chromosome
class Chromosome:
    def __init__(self, genes, fitness):
        self.genes = genes
        self.fitness = fitness

    def get_genes(self):
        return self.genes

    def get_fitness(self):
        return self.fitness

    def set_genes(self, genes):
        self.genes = genes

    def set_fitness(self, fitness):
        self.fitness = fitness


# define the class of the genetic algorithm
def selection(population):
    population_fitness = []
    for i in range(len(population)):
        population_fitness.append(population[i].get_fitness())
    population_fitness = np.array(population_fitness)
    population_fitness = population_fitness / np.sum(population_fitness)
    population_fitness = np.cumsum(population_fitness)
    population_fitness = list(population_fitness)
    population_fitness.insert(0, 0)
    population_fitness.pop()
    population_fitness = np.array(population_fitness)
    population_fitness = population_fitness.tolist()
    population_fitness.append(1)


class GeneticAlgorithm:
    def __init__(self, graph, population_size, crossover_rate, mutation_rate, max_generation):
        self.graph = graph
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generation = max_generation

    def get_graph(self):
        return self.graph

    def get_population_size(self):
        return self.population_size

    def get_crossover_rate(self):
        return self.crossover_rate

    def get_mutation_rate(self):
        return self.mutation_rate

    def get_max_generation(self):
        return self.max_generation

    def set_graph(self, graph):
        self.graph = graph

    def set_population_size(self, population_size):
        self.population_size = population_size

    def set_crossover_rate(self, crossover_rate):
        self.crossover_rate = crossover_rate

    def set_mutation_rate(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def set_max_generation(self, max_generation):
        self.max_generation = max_generation

    # initialize the population
    def initialize_population(self):
        population = []
        for i in range(self.population_size):
            genes = []
            for j in range(len(self.graph.get_matrix())):
                genes.append(random.randint(0, self.graph.get_color() - 1))
            population.append(Chromosome(genes, self.fitness_function(genes)))
        return population

    # the fitness function
    def fitness_function(self, genes):
        fitness = 0
        for i in range(len(self.graph.get_matrix())):
            for j in range(i + 1, len(self.graph.get_matrix())):
                if self.graph.get_matrix()[i][j] == 1 and genes[i] != genes[j]:
                    fitness += 1
        return fitness

    # the crossover operation
    def crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(0, len(parent1.get_genes()) - 1)
            for i in range(crossover_point, len(parent1.get_genes())):
                child1.get_genes()[i] = parent2.get_genes()[i]
                child2.get_genes()[i] = parent1.get_genes()[i]
        return child1, child2

    # the mutation operation
    def mutation(self, child):
        if random.random() < self.mutation_rate:
            mutation_point = random.randint(0, len(child.get_genes()) - 1)
            child.get_genes()[mutation_point] = random.randint(0, self.graph.get_color() - 1)
        return child

    # the evolution operation
    def evolution(self, population):
        new_population = []
        for i in range(len(population)):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        return new_population

    # the main function of the genetic algorithm
    def genetic_algorithm(self):
        population = self.initialize_population()
        best_fitness = []
        for i in range(self.max_generation):
            population = self.evolution(population)
            selection(population)
            best_fitness.append(population[0].get_fitness())
        return best_fitness

    # main function
    def main(self):
        best_fitness = self.genetic_algorithm()
        plt.plot(best_fitness)
        plt.xlabel('generation')
        plt.ylabel('best fitness')
        plt.title('the best fitness of each generation')
        plt.grid()
        plt.savefig('best_fitness.png')
        plt.show()


# the main function
if __name__ == '__main__':
    matrix = [[0, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 1, 1, 0, 0, 0],
              [1, 1, 0, 1, 0, 1, 0, 0, 0],
              [1, 0, 1, 0, 1, 0, 1, 0, 0],
              [0, 1, 0, 1, 0, 1, 1, 0, 0],
              [0, 1, 1, 0, 1, 0, 1, 1, 0],
              [0, 0, 0, 1, 1, 1, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 0]]
    color = 3
    graph = Graph(matrix, color)
    population_size = 100
    crossover_rate = 0.8
    mutation_rate = 0.01
    max_generation = 100
    genetic_algorithm = GeneticAlgorithm(graph, population_size, crossover_rate, mutation_rate, max_generation)
    genetic_algorithm.main()
