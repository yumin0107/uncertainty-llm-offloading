import time
from typing import List, Tuple, Set
import numpy as np
import random

from basestation import User, EdgeServer


def generate_individual(N: int, M: int, edge_only: Set[int]) -> List[int]:
    individual = []
    for i in range(N):
        if i in edge_only:
            gene = random.randint(0, M - 1)
        else:
            gene = random.randint(-1, M - 1)
        individual.append(gene)
    return individual


def crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    point = random.randint(1, len(p1) - 1)  # division point
    c1 = list(np.concatenate([p1[:point], p2[point:]]))
    c2 = list(np.concatenate([p2[:point], p1[point:]]))
    return c1, c2


def mutate(
    ind: List[int], M: int, edge_only: Set[int], rate: float = 0.02
) -> List[int]:
    if random.random() >= rate:
        return ind

    i = random.choice(ind)
    choices = list(range(-1, M))
    if i in edge_only:
        choices.remove(-1)

    choices.remove(ind[i])
    ind[i] = random.choice(choices)
    return ind


def encode_decision(ind: List[int], M: int) -> List[List[int]]:
    N = len(ind)
    mat = [[0] * M for _ in range(N)]
    for i, j in enumerate(ind):
        if j != -1:
            mat[i][j] = 1
    return mat


def fitness(ind: List[int], us: List[User], es: List[EdgeServer], tau: float) -> float:
    decision = encode_decision(ind, len(es))
    total_delta = 0.0
    for i, j in enumerate(ind):
        user = us[i]
        if j == -1:
            total_delta += user.uncertainty * user.t_comp_slm
        else:
            server = es[j]
            C_j_ES = server.compute_allocation(ind.count(j))
            B_j = server.bandwidth_allocation(ind.count(j))
            total_delta += user.uncertainty * (
                server.edge_comp_delay(user, C_j_ES)
                + server.total_comm_delay(user, decision, B_j)
            )
    return -total_delta  # maximize fitness = minimize delay


def select_parents(
    pop: List[List[int]], fits: List[float], num: int
) -> List[List[int]]:
    fits_arr = np.array(fits, dtype=np.float64)
    fits_arr = fits_arr - np.min(fits_arr) + 1e-6
    probs = fits_arr / np.sum(fits_arr)
    pids = list(np.random.choice(len(fits), size=num, replace=False, p=probs))
    return [pop[i] for i in pids]


def ga_offloading(
    us: List[User],
    es: List[EdgeServer],
    tau: float,
    population_size: int = 100,
    generations: int = 5000,
    mutation_rate: float = 0.01,
    elitism_size: int = 6,
    stagnation_limit: int = 200,
) -> Tuple[List[List[int]], int, float]:
    start = time.time()
    N, M = len(us), len(es)
    edge_only: Set[int] = {i for i, u in enumerate(us) if u.uncertainty > tau}

    population = [
        generate_individual(N, M, edge_only) for _ in range(population_size)
    ]  # p x N

    best_ind = None
    best_fit = float("-inf")
    no_improve_count = 0

    for _ in range(generations):
        fits = [fitness(ind, us, es, tau) for ind in population]

        # 최고 해 갱신
        improved = False
        for ind, f in zip(population, fits):
            if f > best_fit:
                best_ind = ind.copy()
                best_fit = f
                improved = True
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= stagnation_limit:
            break

        elite_idxs = np.argsort(fits)[-elitism_size:]
        elites = [population[i].copy() for i in elite_idxs]

        # 부모 선택
        parents = select_parents(population, fits, population_size)

        # 교차 및 돌연변이
        children: List[List[int]] = []
        target_children = population_size - elitism_size
        for i in range(0, len(parents), 2):
            if len(children) >= target_children:
                break
            if i + 1 >= len(parents):
                break
            c1, c2 = crossover(parents[i], parents[i + 1])
            children.append(mutate(c1, M, edge_only, mutation_rate))
            if len(children) < target_children:
                children.append(mutate(c2, M, edge_only, mutation_rate))

        population = elites + children

    end = time.time()
    n_off = sum(val != -1 for val in best_ind)

    for i, j in enumerate(best_ind):
        if j != -1:
            user = us[i]
            server = es[j]
            server.add_user(user)

    for j in range(len(es)):
        e = es[j]
        e.B_j = e.bandwidth_allocation(len(e.users))
        e.C_j_ES = e.compute_allocation(len(e.users))

    return (
        encode_decision(best_ind, M),
        n_off,
        end - start,
    )  # decisions, n_offlaoded, total delay
