__author__ = 'lvova'
class ClusterException(Exception):
    pass


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

    if method in ('ward', 'centroid', 'median'):
        if metrics != 'euclidian':
            raise ClusterException('Метод {} можно использовать только с евклидовым расстоянием'.format(method))
        all_url_ids = _get_space_basis(vectors)  # - список всех различных id, встресающихся в serps
        vectors = [[[(url_id in serp)for url_id in all_url_ids]] for serp in vectors]

    return _get_linkage(vectors, method, metrics)
