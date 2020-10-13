import numpy as np
from overcooked_ai_pcg.helper import lvl_str2grid
from queue import Queue

def bc_demo1(ind):
    return np.random.rand() * 5

def bc_demo2(ind):
    return np.random.rand() * 5

def pot_onion_shortest_dist(ind):
    return shortest_dist('P', 'O', ind.level)

def pot_serve_shortest_dist(ind):
    return shortest_dist('P', 'S', ind.level)

def pot_dish_shortest_dist(ind):
    return shortest_dist('P', 'D', ind.level)

def onion_dish_shortest_dist(ind):
    return shortest_dist('O', 'D', ind.level)

def onion_serve_shortest_dist(ind):
    return shortest_dist('O', 'S', ind.level)

def dish_serve_shortest_dist(ind):
    return shortest_dist('D', 'S', ind.level)

def shortest_dist(terrain1, terrain2, lvl_str):
    """
    Use BFS to find the shortest distance between two specified
    terrain types in the level
    """
    shortest = np.inf
    dxs = (0, 0, 1, -1)
    dys = (1, -1, 0, 0)
    lvl_grid = lvl_str2grid(lvl_str)
    m = len(lvl_grid)
    n = len(lvl_grid[0])
    for i, row in enumerate(lvl_grid):
        for j, terrain in enumerate(row):
            if terrain == terrain1:
                q = Queue()
                seen = set()
                q.put((i, j))
                dist_matrix = np.full((m, n), np.inf)
                dist_matrix[i, j] = 0
                while not q.empty():
                    curr = q.get()
                    x, y = curr
                    if curr in seen:
                        continue
                    if lvl_grid[x][y] == terrain2 and dist_matrix[x, y] < shortest:
                        shortest = dist_matrix[x, y]
                    seen.add(curr)
                    for dx, dy in zip(dxs, dys):
                        n_x = x + dx
                        n_y = y + dy
                        if n_x < m and n_x >= 0 and \
                           n_y < n and n_y >= 0 and \
                           lvl_grid[n_x][n_y] != 'X':
                            q.put((n_x, n_y))
                            dist_matrix[n_x, n_y] = dist_matrix[x, y] + 1
    return shortest