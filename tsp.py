#!/usr/bin/env python3.11

# Copyright 2024, Gurobi Optimization, LLC

# Solve a traveling salesman problem on a randomly generated set of points
# using lazy constraints.  The base MIP model only includes 'degree-2'
# constraints, requiring each node to have exactly two incident edges.
# Solutions to this model may contain subtours - tours that don't visit every
# city.  The lazy constraint callback adds new constraints to cut them off.

import sys
import logging
import math
import random
import time
from collections import defaultdict
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB


def shortest_subtour(edges):
    """Given a list of edges, return the shortest subtour (as a list of nodes)
    found by following those edges. It is assumed there is exactly one 'in'
    edge and one 'out' edge for every node represented in the edge list."""

    # Create a mapping from each node to its neighbours
    node_neighbors = defaultdict(list)
    for i, j in edges:
        node_neighbors[i].append(j)
    assert all(len(neighbors) == 2 for neighbors in node_neighbors.values())

    # Follow edges to find cycles. Each time a new cycle is found, keep track
    # of the shortest cycle found so far and restart from an unvisited node.
    unvisited = set(node_neighbors)
    shortest = None
    while unvisited:
        cycle = []
        neighbors = list(unvisited)
        while neighbors:
            current = neighbors.pop()
            cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for j in node_neighbors[current] if j in unvisited]
        if shortest is None or len(cycle) < len(shortest):
            shortest = cycle

    assert shortest is not None
    return shortest


class TSPCallback:
    """Callback class implementing lazy constraints for the TSP.  At MIPSOL
    callbacks, solutions are checked for subtours and subtour elimination
    constraints are added if needed."""

    def __init__(self, nodes, x):
        self.nodes = nodes
        self.x = x

    def __call__(self, model, where):
        """Callback entry point: call lazy constraints routine when new
        solutions are found. Stop the optimization if there is an exception in
        user code."""
        if where == GRB.Callback.MIPSOL:
            try:
                self.eliminate_subtours(model)
            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                model.terminate()

    def eliminate_subtours(self, model):
        """Extract the current solution, check for subtours, and formulate lazy
        constraints to cut off the current solution if subtours are found.
        Assumes we are at MIPSOL."""
        values = model.cbGetSolution(self.x)
        edges = [(i, j) for (i, j), v in values.items() if v > 0.5]
        tour = shortest_subtour(edges)

        if len(tour) < len(self.nodes):
            # print("No subtours found")
            # add subtour elimination constraint for every pair of cities in tour
            model.cbLazy(
                gp.quicksum(self.x[i, j] for i, j in combinations(tour, 2))
                <= len(tour) - 1
            )


def solve_tsp(nodes, distances):
    """
    Solve a dense symmetric TSP using the following base formulation:

    min  sum_ij d_ij x_ij
    s.t. sum_j x_ij == 2   forall i in V
         x_ij binary       forall (i,j) in E

    and subtours eliminated using lazy constraints.
    """

    with gp.Env() as env, gp.Model(env=env) as m:
        # Create variables, and add symmetric keys to the resulting dictionary
        # 'x', such that (i, j) and (j, i) refer to the same variable.
        x = m.addVars(distances.keys(), obj=distances, vtype=GRB.BINARY, name="e")
        x.update({(j, i): v for (i, j), v in x.items()})

        # Create degree 2 constraints
        for i in nodes:
            m.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == 2)

        # Optimize model using lazy constraints to eliminate subtours
        m.Params.LazyConstraints = 1
        cb = TSPCallback(nodes, x)
        m.optimize(cb)

        # Extract the solution as a tour
        edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
        tour = shortest_subtour(edges)
        assert set(tour) == set(nodes)

        return tour, m.ObjVal


def solve_tsp2(nodes, distances):
    with gp.Env() as env, gp.Model(env=env) as m:
        n = len(nodes)
        x_range = range(1, n + 1)
        x = m.addVars(x_range, x_range, vtype=GRB.BINARY, name="x", lb=0, ub=1)
        u = m.addVars(x_range, vtype=GRB.INTEGER, name="u")
        m.Params.TimeLimit = 120
        m.setObjective(gp.quicksum(distances[i - 1][j - 1] * x[i, j] for i in x_range for j in x_range))
        m.addConstrs(gp.quicksum(x[i, j] for i in x_range if i != j) == 1 for j in x_range)
        m.addConstrs(gp.quicksum(x[i, j] for j in x_range if j != i) == 1 for i in x_range)
        m.addConstrs(u[i] - u[j] + 1 <= (n - 1) * (1 - x[i, j]) for i in x_range for j in x_range if 2 <= i != j <= n)
        m.addConstrs(2 <= u[i] for i in x_range if 2 <= i <= n)
        m.addConstrs(u[i] <= n for i in x_range if 2 <= i <= n)
        m.optimize()
        edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
        # print(edges)
        # print(m.ObjVal)
        return m.ObjVal
        # return tour, m.ObjVal


if __name__ == "__main__":
    N, M = map(int, input().split())
    # N, M = (300, 1)
    npoints = int(N)
    seed = int(M)

    # Create n random points in 2D
    random.seed(seed)
    nodes = list(range(npoints))
    points = [(random.randint(0, 100), random.randint(0, 100)) for i in nodes]

    # Dictionary of Euclidean distance between each pair of points
    distances = {
        (i, j): math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
        for i, j in combinations(nodes, 2)
    }

    t1 = time.time()
    tour, cost = solve_tsp(nodes, distances)
    t1_end = time.time() - t1

    # distances2 = [[math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2))) for i in nodes] for j in nodes]
    # t2 = time.time()
    # cost2 = solve_tsp2(nodes, distances2)
    # t2_end = time.time() - t2

    print("---------------------------------------------------")
    print(f"obj: {cost}")
    print(f"time: {t1_end}")
    # print("---------------------------------------------------")
    # print(f"obj2: {cost2}")
    # print(f"time2: {t2_end}")
