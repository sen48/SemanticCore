"""
На данный момент, реализован только один графофый метод кластеризации - это
Алгоритм кратчайшего незамкнутого пути (КНП)
"""

import functools
import numpy as np
from scipy.spatial.distance import squareform, num_obs_y


def condensed_to_square_index(n, c):
    # converts an index in a condensed array to the
    # pair of observations it represents
    # modified from here: http://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs
    ti = np.triu_indices(n, 1)
    return ti[0][c], ti[1][c]


num_obs = lambda dist: dist.shape[0] if dist.ndim == 2 else num_obs_y(dist)

args_min_dist = lambda dist: condensed_to_square_index(num_obs(dist), _condensed_dist(dist).argmin())

_square_dist = lambda dist: dist if dist.ndim == 2 else squareform(dist)

_condensed_dist = lambda dist: dist if dist.ndim == 1 else squareform(dist)

_get = lambda dist, i, j: dist[i][j] if dist.ndim == 2 else squareform(dist)[i][j]

min_except_zero = lambda lst: functools.reduce(lambda res, x: res if x == 0 else min(res, x), lst, lst[0])


class _Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight


def shotest_open_path(dist, level=0.9):
    """
        Алгоритм кратчайшего незамкнутого пути строит граф из ℓ−1 рёбер так, чтобы
    они соединяли все ℓ точек и обладали минимальной суммарной длиной. Такой граф
    называется кратчайшим незамкнутым путём (КНП), минимальным покрывающим
    деревом или каркасом.
    Алгоритм кратчайшего незамкнутого пути (КНП)
        1: Найти пару точек (i, j) с наименьшим ρij и соединить их ребром;
        2: пока в выборке остаются изолированные точки
        3: найти изолированную точку, ближайшую к некоторой неизолированной;
        4: соединить эти две точки ребром;
        5: удалить рёбра, которые длинее level;
    Parameters
    ----------
    dist: ndarray,
        матрица растояний
    level: numeric,
    Returns
    -------
    clusters: list of lists of int
        список кластеров, здесь под кластером подразумевается список номеров объектов, которые лежат в кластеры.
    """
    dist = _square_dist(dist)
    n, m = dist.shape
    edges = [_Edge(i, j, dist[i][j]) for i in range(n) for j in range(i+1, m) if dist[i][j] <= level]
    edges = sorted(edges, key=lambda e: e.weight)
    isolated = [i for i in range(n)]
    e = edges[0]
    isolated.remove(e.v)
    isolated.remove(e.u)
    clusters = [[e.u, e.v]]
    edges.remove(e)

    while len(isolated) != 0 and len(edges) != 0:
        for i in range(len(edges)):
            if edges[i].u in isolated and edges[i].v not in isolated:
                isolated.remove(edges[i].u)
                clusters[-1].append(edges[i].u)
                edges.remove(edges[i])
                break
            if edges[i].v in isolated and edges[i].u not in isolated:
                isolated.remove(edges[i].v)
                clusters[-1].append(edges[i].v)
                edges.remove(edges[i])
                break
        else:
            for i in range(len(edges)):
                if edges[i].u in isolated and edges[i].v in isolated:
                    isolated.remove(edges[i].u)
                    isolated.remove(edges[i].v)
                    clusters.append([edges[i].u, edges[i].v])
                    edges.remove(edges[i])
                    break
            else:
                break
    for v in isolated:
        clusters.append([v])

    return clusters






