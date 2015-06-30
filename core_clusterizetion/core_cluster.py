import functools
import statistics
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from core_clusterizetion import forel
from core_clusterizetion import graph_metods as gm
import search_query.serps as wrs


class ClusterException(Exception):
    pass


def _get_space_basis(serps):
    #
    all_url = []
    for s in serps:
        for u in s:
            if u not in all_url:
                all_url.append(u)
    return all_url


def _get_linkage(x, method, metric):
    if method == 'ward':
        return sch.linkage(x, method=method, metric=metric)
    y = pdist(x, metric)
    return sch.linkage(y, method=method, metric=metric)


def get_queries_vectors(ya_queries, num_res, is_euclid):

    # Собираем id URLов ТОП{num_res} для всех запросов из списка ya_queries. Другими словами, кадному запросу ставим в
    # соответствие вектор длиной num_res, состоящий из натуральных чисел равных id URLов.
    vectors = [query.get_url_ids(num_res) for query in ya_queries]

    # Так как методы 'ward', 'centroid', 'median' можно использовать только с евклидовым расстоянием, нам нужно
    # изменить структуру векторов запросов, чтобы евклидово расстояние имело смысл. Для этого пронумеруем
    # всевозможные урлы, встречающиеся в выдачах по заданным запросам.
    # Если i-й запрос есть в выдаче по запросу q, то в векторе, соответствующем запросу q на i-м месте стоит 1,
    # иначе  - 0.
    # Евклидово расстояние между двумя такими векторами совпадает с числом 2*(num_res - k),
    # где k - число URLов, присутствующих в обеих выдачах.

    if is_euclid:
        all_url_ids = _get_space_basis(vectors)  # - список всех различных id, встресающихся в serps
        vectors = [[[(url_id in serp)for url_id in all_url_ids]] for serp in vectors]

    return vectors


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
            raise ClusterException('Метод {} можно использовать только с евклидовым расстоянием'.format(method))

    return _get_linkage(get_queries_vectors(ya_queries, num_res, method in ('ward', 'centroid', 'median')), method, metrics)


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
    return renumerate(sch.fcluster(z, level * max(z[:, 2]), criterion='distance'))


def F0(f_cluster, dist):
    N = dist.shape[0]
    if N != len(f_cluster):
        raise ClusterException('len(fcluster) != dist.shape[0]')
    devisor = sum(dist[i][j] for i in range(N) for j in range(i, N))
    dividend = sum(dist[i][j] for i in range(N) for j in range(i, N) if f_cluster[i] == f_cluster[j])
    return dividend / devisor


def F1(f_cluster, dist):
    N = dist.shape[0]
    if N != len(f_cluster):
        raise ClusterException('len(fcluster) != dist.shape[0]')
    devisor = sum(dist[i][j] for i in range(N) for j in range(i, N))
    dividend = sum(dist[i][j] for i in range(N) for j in range(i, N) if f_cluster[i] != f_cluster[j])
    return dividend / devisor


def renumerate(fcl):

    """
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

    pairs = sorted([(k, sum([f == k for f in fcl])) for k in range(1, int(max(fcl)) + 1)], key=lambda x: x[1], reverse=True)
    # создается словарь, где старым номерам класстеров соответствуют новые, в порядке убывания кол-ва эл-тов
    order = {p[0]: i+1 for i, p in enumerate(pairs)}
    # возвращается новый fcl, где старые номера заменяются на новые
    return [order[k] for k in fcl]


if __name__ == "__main__":
    import pandas
    from search_query.ya_query import YaQuery

    def mvp(semcorefile, fout_name, site, region):
        'Kohonen self-organizing map clusterization'
        import mvpa2.suite
        import kohonen.kohonen
        from PIL import ImageDraw
        num_res = 10
        queries = [YaQuery(q, region) for q in wrs.queries_from_file(semcorefile)]
        data = get_queries_vectors(queries,num_res, True)

        data_names = [q.query for q in queries]
        N = 20
        H = 30
        par = kohonen.kohonen.Parameters(
                 dimension=2,
                 shape=(N, H),
                 metric=None,
                 learning_rate=0.05,
                 neighborhood_size=None,
                 noise_variance=None)

        som = kohonen.kohonen.Map(par)
        som.learn(data)
        img = som.distance_heatmap(data)
        img.save("pil-example.png")
        som = mvpa2.suite.SimpleSOMMapper((N, H), 100, learning_rate=0.05)
        som.train(data)
        mapped = som(data)
        mvpa2.suite.pl.title('DATA SOM')
        mvpa2.suite.pl.ylim([0, N])
        mvpa2.suite.pl.xlim([0, H])
        for i, m in enumerate(mapped):
            print(i, m[1], m[0], data_names[i])
            mvpa2.suite.pl.text(m[1], m[0], data_names[i], ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, lw=0))
        mvpa2.suite.pl.savefig('b.png')


    def main(semcorefile, fout_name, site, region):
        num_res = 10
        queries = [YaQuery(q, region) for q in wrs.queries_from_file(semcorefile)]

        # metrics = lambda u, v: competitiveness.rating_dist(u, v, 'classic')
        # metrics = lambda u, v: 1 - float(sum([int(i in v) for i in u])) / num_res
        # metrics = lambda u, v: competitiveness.rating_dist(u, v, 'CTR')
        metrics = lambda u, v: 1 - sum([int(i in v) for i in u]) / num_res
        #z = queries_linkage(queries, 10, 'average', metrics)

        #fcl0 = fcluster(z, 0.95)
        #fcl1 = fcluster(z, 0.9)
        #fcl2 = fcluster(z, 0.8)

        #dist = forel._norm_pdist(pdist([query.get_url_ids(num_res) for query in queries], metrics))
        import pickle
        #pickle.dump(dist, open('dist', mode='wb'))
        dist = pickle.load(open('dist', mode='rb'))

        print('st')
        '''fcl3 = renumerate(forel.forel(dist, 0.9)[0])
        #fcl4 = renumerate(forel.forel(dist, 0.9)[0])
        #fcl5 = renumerate(forel.forel(dist, 0.9)[0])
        for i in range(100):
            print(forel.forel(dist, 0.9)[1])

        fcl =  # [fcl0, fcl1, fcl2, fcl3, fcl4, fcl5]
        for i, f in enumerate(fcl):
            print('{}: F0:{:.4}; F1:{:.4}; N:{}'.format(i+1, F0(f, dist), F1(f, dist), max(f)))'''
        clusters = forel.forel_for_skat(dist, 0.9)[0]
        print(len(clusters))


        def cluster_dist(cl1, cl2, dist1):
            return min(dist1[c1][c2] for c1 in cl1 for c2 in cl2)


        M = len(clusters)

        cdist = np.zeros((M, M))
        for i, cl in enumerate(clusters):
            for j in range(i+1, len(clusters)):
                cdist[i][j] = cluster_dist(cl, clusters[j], dist)
                cdist[j][i] = cdist[i][j]


        clusters_of_cluster_nums = gm.shotest_open_path(cdist, 0.7)
        new_clusters = []
        print('===================================================')
        for cluster_of_nums in clusters_of_cluster_nums:
            new_clusters.append([])
            for num in cluster_of_nums:
                new_clusters[-1] += clusters[num]
                print([queries[i].query for i in clusters[num]])
            print('+++++++++++++++++++++++++++++++++')
        print('===================================================')
        print(M)
        print(len(clusters_of_cluster_nums))








        #print_clusters(fcl, queries, site, fout_name)
        #plot_dendrogram(z, fcl=fcl[0], fig=1, labels=[q.query for q in queries])

    def print_clusters(fcls, queries, site, report_file):

        """

        :param fcls: list
        :param queries: список объектов типа YaQuery
        :param report_file: фаил, в который пишем результат
        """

        fcl_cols = ['cl_{}'.format(i) for i, f in enumerate(fcls)]
        columns = ['запрос']
        columns += fcl_cols
        columns += ['соотв позиция', 'соотв стр']
        ps = [0 for query in queries]  # [wrs.read_url_position(site, query.query,  query.region) for query in queries]
        pos = [p for p in ps]  # [p[0] for p in ps]
        pgs = [p for p in ps]  # [p[1].url for p in ps]
        array = [[query.query for query in queries]]
        array += fcls
        array += [pos, pgs]
        data_frame = pandas.DataFrame(np.array(array).T, columns=columns)
        wrs.write_report(data_frame.sort(fcl_cols.append('соотв стр')), report_file)

    main('C:\\_Work\\lightstar\\to_filter.txt', 'c:\\_Work\\lightstar\\result_clust_4.csv', 'lightstar.ru', 213)
    #mvp('C:\\_Work\\lightstar\\to_clust.txt', 'c:\\_Work\\trav\\result_clust_2.csv', 'newlita.com', 2)