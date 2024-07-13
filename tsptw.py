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


def solve_tsptw(k, distance, tw):
    def call_back(model: gp.Model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                print(model.cbGetSolution())
            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                model.terminate()

    with gp.Env() as env, gp.Model(env=env) as m:
        var_range = range(0, 2 * k + 1)
        x = m.addVars(var_range, var_range, vtype=GRB.BINARY, name="x")
        t = m.addVars(var_range, vtype=GRB.CONTINUOUS, name="t")
        obj = m.addVar(vtype=GRB.CONTINUOUS, name="obj")

        # Create degree 2 constraints
        for i in var_range:
            m.addConstr(gp.quicksum(x[i, j] for j in var_range if i != j) == 1)

        for j in var_range:
            m.addConstr(gp.quicksum(x[i, j] for i in var_range if i != j) == 1)

        m.addConstr(t[0] == 0)
        m.addConstrs(t[i] + distance[i][j] - t[j] <= 1000000 * (1 - x[i, j]) for i in var_range for j in range(1, 2*k+1) if i != j)
        # m.addConstrs(t[i] + 1 - t[j] <= 1000000 * (1 - x[i, j]) for i in var_range for j in var_range if i != j)

        for i in range(1, k + 1):
            m.addConstr(tw[i - 1][0] <= t[i])

        for i in range(1, k + 1):
            for j in range(k+1, 2*k + 1):
                m.addConstr(t[i] <= t[j])


        for i in range(k + 1, 2 * k + 1):
            m.addConstr(t[i] <= tw[i - (k + 1)][1])

        for i in var_range:
            m.addConstr(t[i] <= obj)

        # m.Params.LazyConstraints = 1
        # Optimize model using lazy constraints to eliminate subtours
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()
        print("-------------------------")
        print("-------------------------")
        print(tw)
        print(obj)
        print([(i, v.X) for i, v in t.items()])
        print([(i, j) for (i, j), v in x.items() if v.X > 0.5])
        return m.ObjVal


if __name__ == "__main__":
    # N, M = map(int, input().split())
    # N, M = (300, 1)
    # npoints = int(N)
    # seed = int(M)

    # Create n random points in 2D
    # random.seed(seed)
    # nodes = list(range(npoints))
    # points = [(random.randint(0, 100), random.randint(0, 100)) for i in nodes]

    # Dictionary of Euclidean distance between each pair of points
    # distances = {
    #     (i, j): math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
    #     for i, j in combinations(nodes, 2)
    # }
    distance = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 120.0, 474.0, 474.0, 320.0, 389.0, 560.0, 513.0, 362.0],
                [0, 474.0, 120.0, 120.0, 375.0, 229.0, 317.0, 271.0, 253.0],
                [0, 474.0, 120.0, 120.0, 375.0, 229.0, 317.0, 271.0, 253.0],
                [0, 320.0, 375.0, 375.0, 120.0, 266.0, 539.0, 487.0, 244.0],
                [0, 389.0, 229.0, 229.0, 266.0, 120.0, 405.0, 353.0, 148.0],
                [0, 560.0, 317.0, 317.0, 539.0, 405.0, 120.0, 173.0, 419.0],
                [0, 513.0, 271.0, 271.0, 487.0, 353.0, 173.0, 120.0, 366.0],
                [0, 362.0, 253.0, 253.0, 244.0, 148.0, 419.0, 366.0, 120.0]]
    tw = [[0, 1716],
          [469, 2095],
          [380, 1949],
          [755, 2290]]




    k = 4
    t1 = time.time()
    cost = solve_tsptw(k, distance, tw)
    t1_end = time.time() - t1

    print("---------------------------------------------------")
    print(f"obj: {cost}")
    print(f"time: {t1_end}")
