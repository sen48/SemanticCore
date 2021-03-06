"""
    Комбинация алгоритмов ФОРЭЛ и КНП.
    Сначала строится кластеризация алгоритмом ФОРЭЛ, далее на вход алгоритму кратчайшего незамкнутого пути подаются
    вместо объектов кластеры ФОРЭЛ. Кластеры ФОРЭЛ, попавшие в один кластер КНП объединяются.
"""
from scipy.spatial.distance import pdist
import numpy as np

import core_clusterizetion.forel as forel
import core_clusterizetion.graph_metods as gr
import core_clusterizetion.core_cluster as core_cluster
from search_engine_tk.ya_query import queries_from_file


def forel_shotest_open_path(dist, Rforel, Rsop):

    """
    Комбинация алгоритмов ФОРЭЛ и КНП.
    Сначала строится кластеризация алгоритмом ФОРЭЛ, далее на вход алгоритму кратчайшего незамкнутого пути подаются
    вместо объектов кластеры ФОРЭЛ. Кластеры ФОРЭЛ, попавшие в один кластер КНП объединяются.

    Parameters
    ----------
    dist: ndarray,
        матрица растояний
    Rforel: numeric,
        радиус поиска локальных сгущений для ФОРЭЛ
    Rsop: numeric,
        уровень для КНП

    Returns
    -------
    : ndarray
        Массив длины равной количеству кластеризуемых запросов. i - й элемент этого списка равен номеру класстера,
        к которому отнесен i - й запрос. Класстеры пронумерованы в порядке убывания количества запросов.
    """
    clusters = forel.forel_for_skat(dist, Rforel)[0]

    m = len(clusters)
    cdist = np.zeros((m, m))

    for i, cl in enumerate(clusters):
        for j in range(i+1, len(clusters)):
            cdist[i][j] = core_cluster.cluster_dist(cl, clusters[j], dist)
            cdist[j][i] = cdist[i][j]
    clusters_of_cluster_nums = gr.shotest_open_path(cdist, Rsop)

    new_clusters = []
    for cluster_of_nums in clusters_of_cluster_nums:
        new_clusters.append([])
        for num in cluster_of_nums:
            new_clusters[-1] += clusters[num]
    fcluster = [0 for q in queries]
    for i, cluster in enumerate(new_clusters):
        for obj in cluster:
            fcluster[obj] = i+1
    return core_cluster.renumerate(fcluster)


if __name__ == "__main__":
    region = 213
    semcorefile = 'C:\\_Work\\vostok\\to_clust.txt'
    num_res = 10
    Rforel = 0.9

    queries = queries_from_file(semcorefile, region)
    vectors = core_cluster.get_queries_vectors(queries, num_res, False)

    print('done')
    metrics = lambda u, v: 1 - sum([int(i in v) for i in u]) / num_res
    y = pdist(vectors, metrics)

    fcl1 = forel_shotest_open_path(y, 0.9, 0.6)
    fcl2 = forel_shotest_open_path(y, 0.9, 0.7)
    core_cluster.print_clusters([fcl1, fcl2], queries, 'c:\\_Work\\vostok\\res_4.csv', '')

