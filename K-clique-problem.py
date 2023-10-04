# code the clique problem by breaking the graph into subgraphs in oop

import matplotlib.pyplot as plt

from main import GeneticAlgorithm


# define the class of the graph
def get_subgraph_size(subgraph):
    return len(subgraph)


class Graph:
    def __init__(self, matrix, color):
        self.population_size = None
        self.max_generation = None
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

    def get_subgraph(self, subgraph):
        subgraph_matrix = []
        for i in range(len(subgraph)):
            subgraph_matrix.append([])
            for j in range(len(subgraph)):
                subgraph_matrix[i].append(self.matrix[subgraph[i]][subgraph[j]])
        return Graph(subgraph_matrix, self.color)

    def get_subgraph_color(self, subgraph):
        subgraph_color = []
        for i in range(len(subgraph)):
            subgraph_color.append(self.color[subgraph[i]])
        return subgraph_color

    def get_subgraph_edges(self, subgraph):
        subgraph_edges = 0
        for i in range(len(subgraph)):
            for j in range(len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1:
                    subgraph_edges += 1
        return subgraph_edges / 2

    def get_subgraph_edges2(self, subgraph):
        subgraph_edges = 0
        for i in range(len(subgraph)):
            for j in range(len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1:
                    subgraph_edges += 1
        return subgraph_edges

    def get_subgraph_edges3(self, subgraph):
        subgraph_edges = 0
        for i in range(len(subgraph)):
            for j in range(len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1:
                    subgraph_edges += 1
        return subgraph_edges / 2

    def get_subgraph_edges4(self, subgraph):
        subgraph_edges = 0
        for i in range(len(subgraph)):
            for j in range(len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1:
                    subgraph_edges += 1
        return subgraph_edges / 2

    def get_subgraph_edges5(self, subgraph):
        subgraph_edges = 0
        for i in range(len(subgraph)):
            for j in range(len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1:
                    subgraph_edges += 1
        return subgraph_edges / 2

    # define the class of the chromosome
    def get_subgraph_fitness(self, subgraph):
        subgraph_fitness = 0
        for i in range(len(subgraph)):
            for j in range(i + 1, len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1 and self.color[subgraph[i]] != self.color[subgraph[j]]:
                    subgraph_fitness += 1
        return subgraph_fitness

    def get_subgraph_fitness2(self, subgraph):
        subgraph_fitness = 0
        for i in range(len(subgraph)):
            for j in range(i + 1, len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1 and self.color[subgraph[i]] != self.color[subgraph[j]]:
                    subgraph_fitness += 1
        return subgraph_fitness

    def get_subgraph_fitness3(self, subgraph):
        subgraph_fitness = 0
        for i in range(len(subgraph)):
            for j in range(i + 1, len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1 and self.color[subgraph[i]] != self.color[subgraph[j]]:
                    subgraph_fitness += 1
        return subgraph_fitness

    def get_subgraph_fitness4(self, subgraph):
        subgraph_fitness = 0
        for i in range(len(subgraph)):
            for j in range(i + 1, len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1 and self.color[subgraph[i]] != self.color[subgraph[j]]:
                    subgraph_fitness += 1
        return subgraph_fitness

    def get_subgraph_fitness5(self, subgraph):
        subgraph_fitness = 0
        for i in range(len(subgraph)):
            for j in range(i + 1, len(subgraph)):
                if self.matrix[subgraph[i]][subgraph[j]] == 1 and self.color[subgraph[i]] != self.color[subgraph[j]]:
                    subgraph_fitness += 1
        return subgraph_fitness

    # define the plot function
    def plot(self):
        plt.figure()
        plt.imshow(self.matrix)
        plt.show()

    def get_subgraph_size(self):
        pass

    def selection(self, population):
        pass

    def crossover(self, param, param1):
        pass

    def mutation(self, param):
        pass

    def survival_selection(self, population):
        pass


# define the main function
def main():
    print("K-clique problem")
    # define the graph
    matrix = [[0, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 1, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0, 1, 1, 1, 0],
              [0, 0, 1, 1, 1, 0, 1, 1, 1],
              [0, 0, 0, 1, 1, 1, 0, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 0, 1],
              [0, 0, 0, 0, 0, 1, 1, 1, 0]]
    color = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    graph = Graph(matrix, color)
    # plot the graph
    graph.plot()
    # define the parameters
    population_size = 1000
    crossover_rate = 0.8
    mutation_rate = 2
    max_generation = 100
    # define the genetic algorithm
    genetic_algorithm = GeneticAlgorithm(graph, population_size, crossover_rate, mutation_rate, max_generation)
    # run the genetic algorithm
    best_fitness = genetic_algorithm.genetic_algorithm()
    # plot the best fitness
    plt.figure()
    plt.plot(best_fitness)
    plt.xlabel('generation')
    plt.ylabel('best fitness')
    plt.title('the best fitness of each generation')
    plt.grid()
    plt.savefig('best_fitness.png')
    plt.show()


if __name__ == '__main__':
    main()
