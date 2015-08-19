import scipy.cluster.hierarchy as sch
import core_clusterizetion.core_cluster as core_cluster
from scipy.spatial.distance import pdist


class AlgomerativeClusterizationException(Exception):
    pass


def fcluster(z, level):
    """

    Forms flat clusters from the hierarchical clustering defined by
    the linkage matrix ``Z``.

    Parameters
    ----------
    Z : ndarray
        Иерархическая кластеризация представленная в виде linkage matrix, полученная в результате применения функции
        `queries_linkage`.
    level : float
        Доля от максимального расстояния между кластерами. maxdist * level - пороговое значение расстояния, то есть
        если расстояние между кластерами больше этого значения, то дальше кластеры не объединяются.
    Returns
    -------
    fcluster : ndarray
        Массив длины равной количеству кластеризуемых запросов. i - й элемент этого списка равен номеру класстера,
        к которому отнесен i - й запрос. Класстеры пронумерованы в порядке убывания количества запросов.
    """
    return core_cluster.renumerate(sch.fcluster(z, level * max(z[:, 2]), criterion='distance'))


def _get_linkage(x, method, metric):
    if method == 'ward':
        return sch.linkage(x, method=method, metric=metric)
    y = pdist(x, metric)
    return sch.linkage(y, method=method, metric=metric)


def queries_linkage(ya_queries, num_res, method, metrics):
    """
    Функция возвращяет матрицу 'Z' (linkage matrix), число строк которой равно len(queries)-1, а число столбцов 4.
    На i`-ой итерации кластеры с номерами Z[i, 0] и Z[i, 1] объединяются в кластер с индексом n + i.
    Кластер с номером j, меньшим len(queries) соответствует запросу queries[j].
    Расстояние между кластерами Z[i, 0] и Z[i, 1] равно Z[i, 2]``.
    4-й столбец Z[i, 3] содержит число запросов в кластере, полученном в результате объединения.

    Методы расчета расстояния между свежесформированным кластером 'u' и произвольным кластером `v`:
    * method='single':
           d(u,v) = min(dist(u[i],v[j]))
    * method='complete':
           d(u, v) = max(dist(u[i],v[j]))
    * method='average':
           d(u,v) = sum_{ij}( d(u[i], v[j])/(|u|*|v|))
    * method='weighted' assigns
           d(u,v) = (dist(s,v) + dist(t,v))/2,
        где кластер u получен при объединении кластеров s и t.
    * method='centroid':
           d(u,v) = dist((c_s + c_t) / 2, c_v)
        где кластер u получен при объединении кластеров s и t,
        c_u и c_v центры кластеров u и v, соответственно.
    * method='median':
            d(u,v) = dist(c_u,c_v)
        c_u и c_v центры кластеров u и v, соответственно. Здесь c_W = sum_{w in W}( w/|W|, |*| - мощность множества.

    * method='ward'
            d(u,v) = sqrt{(|v|+|s|)/T * d(v,s)^2  + (|v|+|t|)/T * d(v,t)^2  + |v|/T * d(s,t)^2},
        где `T=|v|+|s|+|t|`, и |*| - мощность множества
    Методы 'ward', 'centroid', 'median' можно использовать только с евклидовым расстоянием,

    Parameters
    ----------
    ya_queries: tuple or list of  YaQuery
        Поисковые запросы
    num_res: int
        Глубина ТОПа
    method : str
        Метод расчета расстояния между класстерами при иерархической кластеризации.
    metric : str or function
        С помощью этои метрики расчитывается расстояние между векторами запросов
    Returns
    -------
    Z : ndarray
        Иерархическая кластеризация представленная в виде linkage matrix.
    """

    if method in ('ward', 'centroid', 'median'):
        if metrics != 'euclidian':
            raise AlgomerativeClusterizationException('Метод {} можно использовать '
                                                      'только с евклидовым расстоянием'.format(method))

    vectors = core_cluster.get_queries_vectors(ya_queries, num_res, method in ('ward', 'centroid', 'median'))
    return _get_linkage(vectors, method, metrics)

if __name__ == "__main__":
    from search_engine_tk.ya_query import queries_from_file
    import core_clusterizetion.visual as visual

    def main(semcorefile, fout_name, site, region):
        num_res = 10
        queries = queries_from_file(semcorefile, region)

        metrics = lambda u, v: 1 - sum([int(i in v) for i in u]) / num_res
        print('reading SERPs')
        serps = [query.get_serp(num_res) for query in queries]
        print(' done')

        ids = [[item.url_id for item in s] for i, s in enumerate(serps)]
        dist = pdist([i for i in ids], metrics)

        z = sch.linkage(dist, method='average', metric=metrics)
        print('Linkage OK')

        fcl0 = fcluster(z, 0.95)
        fcl1 = fcluster(z, 0.9)
        fcl2 = fcluster(z, 0.8)
        fcl3 = fcluster(z, 0.7)
        fcl4 = fcluster(z, 0.6)
        fcl5 = fcluster(z, 0.5)
        print('Algomerative ok')

        fcls = [fcl0, fcl1, fcl2, fcl3, fcl4, fcl5]

        core_cluster.print_clusters(fcls, queries, fout_name, site)
        print('Print Ok')
        visual.plot_dendrogram(z, fcl=fcl0, fig=1, labels=[q.query for q in queries])

    main('C:\\_Work\\vostok\\to_clust_prav.txt', 'c:\\_Work\\vostok\\result_clust_4.csv', None, 213)