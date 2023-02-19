from typing import Tuple, Union

import numpy as np


def make_initial_population(t_size: Tuple[int, int],
                            t_low: float = -4.0,
                            t_high: float = 4.0) -> np.ndarray:
    """Makes the initial population. Populated randomly.

    Args:
        t_size (Tuple[int,int]): Size of the population.
        t_low (float, optional): Lower limit of uniform distribution. Defaults to -4.0.
        t_high (float, optional): Upport limit of the uniform distribution. Defaults to 4.0.

    Returns:
        np.ndarray: _description_
    """
    return np.random.uniform(low=t_low, high=t_high, size=t_size)


def mutate(t_offspring: np.ndarray) -> np.ndarray:
    """Changes the gene in each offspring randomly. Uniform mutation is implementeed.

    Args:
        t_offspring (np.ndarray): offspring subjected to mutation

    Returns:
        np.ndarray: mutated form of the offspring
    """
    for idx in range(t_offspring.shape[0]):
        random_value = np.random.uniform(-1., 1., 1)
        t_offspring[idx, 4] = t_offspring[idx, 4] + random_value
    return t_offspring


def crossover(t_survivor: np.ndarray,
              t_size: Tuple[int, int]) -> np.ndarray:
    """Half of the first parent's gene is recombined with
    half of the second parent's gene which is defined in the 
    crossover point. One-point crossover is implemente.

    Args:
        t_survivor (np.ndarray): parents
        t_size (Tuple[int,int]): population size

    Returns:
        np.ndarray: crossover result (offspring)
    """
    offspring = np.empty(t_size)
    crossover_point = np.uint8(t_size[1]/2)

    for i in range(t_size[0]):
        first_parent_idx = i % t_survivor.shape[0]
        second_parent_idx = (i+1) % t_survivor.shape[0]
        offspring[i, 0:crossover_point] = t_survivor[first_parent_idx,
                                                     0:crossover_point]
        offspring[i, crossover_point:] = t_survivor[second_parent_idx,
                                                    crossover_point:]

    return offspring


def evaluate_fitness(t_eq: Tuple[float, ...],
                     t_population: np.ndarray) -> np.ndarray:
    """Evaluate the fitness score of the population

    Args:
        t_eq (Tuple[float, ...]): equation to be optimized
        t_population (np.ndarray): pool of candidate solutions

    Returns:
        np.ndarray: fitness score of the population
    """
    return np.sum(t_population * t_eq, axis=1)


def make_selection(t_population: np.ndarray,
                   t_fitness_val: Union[float, np.ndarray],
                   t_parents: int) -> np.ndarray:
    """Selection process in the GA. Filters the 
    candidate solution according to their proximity to the 
    defined solution.

    Args:
        t_population (np.ndarray): pool of candidate solutions
        t_fitness_val (float): fitness value to be optimized
        t_parents (int): parents

    Returns:
        np.ndarray: filtered pool of candidates that represents the next generation
    """
    survivors = np.empty((t_parents, t_population.shape[1]))
    for idx in range(t_parents):
        max_fitness_idx = np.where(t_fitness_val == np.max(t_fitness_val))
        max_fitness_idx = max_fitness_idx[0][0]
        survivors[idx, :] = t_population[max_fitness_idx, :]
        t_fitness_val[max_fitness_idx] = -np.iinfo(np.int32).min

    return survivors


if __name__ == '__main__':
    eq_inputs = (4, -2, 3.5, 5, -11, -4.7)
    weights = 6
    pool_size = 8
    parent_donors = 4
    pop_size = (pool_size, weights)
    pop = make_initial_population(t_size=pop_size)
    generations = 20

    for gen in range(generations):
        # print(f'Generation: {gen}')
        fitness = evaluate_fitness(eq_inputs, pop)
        parents = make_selection(pop, fitness, parent_donors)
        offspring_cro = crossover(parents, t_size=(
            pop_size[0]-parents.shape[0], weights))
        offspring_mut = mutate(offspring_cro)

        pop[0:parents.shape[0], :] = parents
        pop[parents.shape[0]:, :] = offspring_mut

        # print(f'Best result: {np.max(np.sum(pop*eq_inputs, axis=1))}')

    fitness = evaluate_fitness(eq_inputs, pop)
    best_match_idx = np.where(fitness == np.max(fitness))
    print(f'Best solution: {pop[best_match_idx,:]}')
    print(f'Best solution fitness: {fitness[best_match_idx]}')
