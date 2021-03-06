"""
В модуле собрано несколько функций. нужных для работы с результатами кластеризации, независимо от метода кластеризации

F0(Среднее внутриклассовое расстояние) и F1(Среднее межклассовое расстояние) - это метрики качества кластеризации.

"""

import numpy as np
from scipy.spatial.distance import squareform


def square_pdist(pdist):
    """
    Независимо от того в какой форме передана матрица расстояний, возвращяет ее в квадратной форме.

    Parameters
    ----------
        dist: ndarray, матрица расстояний между объектами, может быть как в квадратном, так и в конденсированном
             (condensed) виде
    Returns
    -------
        dist: ndarray, квадратная матрица расстояний между объектами,
    """
    return squareform(pdist) if pdist.ndim == 1 else pdist


def get_queries_vectors(ya_queries, num_res, is_euclid):
    """
    По запросам подгатавливаются векторы, на основе расстояний между этими векторами будет провадиться кластеризация.
    Собираем id URLов ТОП{num_res} для всех запросов из списка ya_queries. Другими словами, кадному запросу ставим в
    соответствие вектор длиной num_res, состоящий из натуральных чисел равных id URLов.

    Так как методы 'ward', 'centroid', 'median' можно использовать только с евклидовым расстоянием, в этом случае
    вышеописанные векторы использовать недьзя и нужно изменить структуру векторов запросов, чтобы евклидово расстояние
    имело смысл. Для этого пронумеруем всевозможные урлы, встречающиеся в выдачах по заданным запросам.
    Если i-й запрос есть в выдаче по запросу q, то в векторе, соответствующем запросу q на i-м месте стоит 1,
    иначе  - 0.
    Евклидово расстояние между двумя такими векторами совпадает с числом 2*(num_res - k),
    где k - число URLов, присутствующих в обеих выдачах.

    Parameters
    ----------
        ya_queries:  list of YaQuery objects, список поисковых запросов
        num_res: int, глубина ТОПа
        is_euclid: bool, истина когда нам нужны векторы, которые можно использовать с евклидовым расстоянием

    Returns
    -------
        vectors: list of int,
    """

    vectors = [query.get_url_ids(num_res) for query in ya_queries]
    if is_euclid:
        all_url_ids = _get_space_basis(vectors)  # - список всех различных id, встресающихся в serps
        vectors = [[[(url_id in serp) for url_id in all_url_ids]] for serp in vectors]

    return vectors


def _get_space_basis(serps):
    """
    Список различных url (url_id) в поисковых выдачах serps, без повторений

    Parameters
    ----------
        serps: list of lists of int or str

    Returns
    -------
        all_url: list of int or str
    """
    all_url = []
    for s in serps:
        for u in s:
            if u not in all_url:
                all_url.append(u)
    return all_url


def cluster_dist(first_cluster, second_cluster, dist):
    """
    Расстояние между кластерами, считается по методу ближнего соседа.
    d(u,v) = min(dist(u[i],v[j]))

    Parameters
    ----------
        first_cluster: list of int, список номеров объектов первого кластера
        second_cluster: list of int, список номеров объектов второго кластера
        dist: ndarray, квадратная матрица расстояний между объектами,

    Returns
    -------
        cluster_dist: float, расстояние между кластерами
    """
    dist = square_pdist(dist)
    return min(dist[c1][c2] for c1 in first_cluster for c2 in second_cluster)


class _ClusterException(Exception):
    pass


def F0(f_cluster, dist):
    """
    Среднее внутриклассовое расстояние. Хорошо, когда оно маленькое.

    Parameters
    ----------
        f_cluster: list of int, f_cluster[i] - номер кластера, к которому отнесен i-й объект.
        dist: ndarray, квадратная матрица расстояний между объектами,

    Returns
    -------
        F0: float, среднее внутриклассовое расстояние
    """
    N = dist.shape[0]
    if N != len(f_cluster):
        raise _ClusterException('len(fcluster) != dist.shape[0]')
    devisor = sum(dist[i][j] for i in range(N) for j in range(i, N))
    dividend = sum(dist[i][j] for i in range(N) for j in range(i, N) if f_cluster[i] == f_cluster[j])
    return dividend / devisor


def F1(f_cluster, dist):
    """
    Среднее межклассовое расстояние. Хорошо, когда оно большое.

    Parameters
    ----------
        f_cluster: list of int, f_cluster[i] - номер кластера, к которому отнесен i-й объект.
        dist: ndarray, квадратная матрица расстояний между объектами,

    Returns
    -------
        F0: float, среднее межклассовое расстояние
    """
    N = dist.shape[0]
    if N != len(f_cluster):
        raise _ClusterException('len(fcluster) != dist.shape[0]')
    devisor = sum(dist[i][j] for i in range(N) for j in range(i, N))
    dividend = sum(dist[i][j] for i in range(N) for j in range(i, N) if f_cluster[i] != f_cluster[j])
    return dividend / devisor


def renumerate(fcl):
    """
    Перенумеровывает кластеры так, чтобы в кластере с меньшим номером содержалось больше объектов

    Parameters
    ----------
        fcl : ndarray
            Массив длины равной количеству кластеризуемых запросов. i - й элемент этого списка равен номеру класстера,
            к которому отнесен i - й запрос. Класстеры пронумерованы в порядке объединения
    Returns
    -------
        fcluster : ndarray
            Массив длины равной количеству кластеризуемых запросов. i - й элемент этого списка равен номеру класстера,
            к которому отнесен i - й запрос. Класстеры пронумерованы в порядке убывания количества запросов.
    """

    # создается список пар (номер класстера, количество элементов в этом класстере) для всех номеров класстеров
    # и упорядочевается по убыванию кол-ва элементов

    pairs = sorted([(k, sum([f == k for f in fcl])) for k in range(1, int(max(fcl)) + 1)], key=lambda x: x[1],
                   reverse=True)
    # создается словарь, где старым номерам класстеров соответствуют новые, в порядке убывания кол-ва эл-тов
    order = {p[0]: i + 1 for i, p in enumerate(pairs)}
    # возвращается новый fcl, где старые номера заменяются на новые
    return [order[k] for k in fcl]


def print_clusters(fcls, queries, report_file, site=None):
    """
    Записывает в фаил результаты кластеризаций, если  site != None, то дописывает релевантные страницы и позиции в
    выдаче

    Parameters
    ----------
        fcls: list списое результатов кластеризации
        queries: список объектов типа YaQuery
        report_file: фаил, в который пишем результат
    """

    import pandas
    import search_engine_tk.serp as wrs

    fcl_cols = ['cl_{}'.format(i) for i, f in enumerate(fcls)]
    columns = ['запрос']
    columns += fcl_cols
    array = [[]] * (len(fcls) + 1)
    array[0] = [query.query for query in queries]
    array[1: 1 + len(fcls)] = fcls
    if site:
        columns += ['соотв позиция', 'соотв стр']
        ps = [wrs.read_url_position(site, query.query,  query.region) for query in queries]
        pos = [p[0] for p in ps]
        pgs = [p[1].url for p in ps]
        array += [[], []]
        array[1 + len(fcls): 3 + len(fcls)] = [pos, pgs]
    data_frame = pandas.DataFrame(np.array(array).T, columns=columns)
    data_frame.sort(fcl_cols.append('соотв стр')).to_csv(report_file, sep=';', index=False)
