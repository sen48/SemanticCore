import numpy as np
import search_query.serps as wrs
from scipy.spatial.distance import squareform


norm_pdist = lambda pdist: squareform(pdist) if pdist.ndim == 1 else pdist


def _get_space_basis(serps):
    #
    all_url = []
    for s in serps:
        for u in s:
            if u not in all_url:
                all_url.append(u)
    return all_url


def cluster_dist(first_cluster, second_cluster, dist):

    """
    Расстояние между кластерами
    :param first_cluster: список номеров объектов первого кластера
    :param second_cluster: список номеров объектов второго кластера
    :param dist: матрица рвсстояний между объеклами
    :return:
    """
    dist = norm_pdist(dist)
    return min(dist[c1][c2] for c1 in first_cluster for c2 in second_cluster)


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



class ClusterException(Exception):
    pass


def F0(f_cluster, dist):
    """
    Среднее внутриклассовое расстояние
    """
    N = dist.shape[0]
    if N != len(f_cluster):
        raise ClusterException('len(fcluster) != dist.shape[0]')
    devisor = sum(dist[i][j] for i in range(N) for j in range(i, N))
    dividend = sum(dist[i][j] for i in range(N) for j in range(i, N) if f_cluster[i] == f_cluster[j])
    return dividend / devisor


def F1(f_cluster, dist):
    """
    Среднее межклассовое расстояние
    """
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


def print_clusters(fcls, queries, site,report_file):

        """

        :param fcls: list
        :param queries: список объектов типа YaQuery
        :param report_file: фаил, в который пишем результат
        """
        import pandas
        fcl_cols = ['cl_{}'.format(i) for i, f in enumerate(fcls)]
        columns = ['запрос']
        columns += fcl_cols
        columns += ['соотв позиция', 'соотв стр']
        ps = [0 for query in queries]  # [wrs.read_url_position(site, query.query,  query.region) for query in queries]
        pos = [p for p in ps]  # [p[0] for p in ps]
        pgs = [p for p in ps]  # [p[1].url for p in ps]
        array = [[] for i in range(len(fcls)+3)]
        array[0] = [query.query for query in queries]
        array[1: 1+len(fcls)] = fcls
        array[1+len(fcls): 3+len(fcls)] = [pos, pgs]
        data_frame = pandas.DataFrame(np.array(array).T, columns=columns)
        wrs.write_report(data_frame.sort(fcl_cols.append('соотв стр')), report_file)


if __name__ == "__main__":
    import pandas
    from search_query.ya_query import YaQuery

    def mvp(semcorefile, fout_name, site, region):
        '''Kohonen self-organizing map clusterization

        '''

        import mvpa2.suite
        import kohonen.kohonen
        from search_query.ya_query import queries_from_file
        num_res = 10
        queries = queries_from_file(semcorefile, region)
        data = get_queries_vectors(queries, num_res, True)

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





    #mvp('C:\\_Work\\lightstar\\to_clust.txt', 'c:\\_Work\\trav\\result_clust_2.csv', 'newlita.com', 2)