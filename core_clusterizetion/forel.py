"""
Алгоритмы кластеризации ФОРЕЛ и СКАТ. О данных методах моэно почитать, например, в
Загоруйко Н. Г. Прикладные методы анализа данных и знаний.
"""

import random
import numpy as np
import core_clusterizetion.core_cluster as cc


def center_of_objects(pdist, neighbour_objects):
    """ возвращает центр тяжести neighbour_objects
        В метрическом пространстве — объект, сумма расстояний до которого минимальна, среди всех внутри сферы
    :param pdist: квадратная матрица растояний между всеми объектами задачи
    :param neighbour_objects:  список номеров объектов для которых нужно найти объект, являющейся центром тяжести
    """

    sum_d = np.array([sum(pdist[i][j] for j in neighbour_objects) for i in neighbour_objects])
    return neighbour_objects[sum_d.argmin()]


def n_times_forel(dist, radius, n_tries=200):
    """
    n_tries  раз запускается алгоритм ФОРЕЛ (FOREL)
    На выходе результат попытки с наименьшей суммой внутриклассовых дисперсий.
    :type n_tries: число  раз запуска алгоритма.
    :param dist: матрица расстояний между объектами
    :param radius: радиус поиска локальных сгущений
    :return: f_cluster[i] - номер кластера, к которому отнесен i-й объект.
            quality - сумма внутриклассовых дисперсий для разбиения f_cluster.
     """

    dist = cc.square_pdist(dist)
    f_cluster = np.ones(dist.shape[0])
    quality = None
    f_centers = []
    for i in range(n_tries):
        clusters, centers = forel(dist, radius)
        q = F(centers, clusters, dist)
        if not quality or q < quality:
            quality = q
            f_cluster = clusters
            f_centers = centers
    return f_cluster, quality, f_centers


def forel_for_skat(dist, radius, n_tries=200):
    """
    То же что и forel, только результаты преобразованы для skat
    """
    f_cluster, quality, centers = n_times_forel(dist, radius, n_tries)

    clusters = [[]] * int(max(f_cluster))
    for i_q, k in enumerate(f_cluster):
        clusters[int(k)-1].append(i_q)

    return clusters, centers


def forel(dist, radius):
    """
    алгоритм ФОРЕЛ (FOREL)
    :param dist: матрица расстояний между объектами
    :param radius:  радиус поиска локальных сгущений
    :return: clusters[i] - номер кластера, к которому отнесен i-й объект.
             centers[j] - номер объекта, являющегося центром j-го кластера.
    """
    objects = [i for i in range(dist.shape[0])]
    cur_num = 1
    clusters = np.ones(dist.shape[0])
    centers = []
    while len(objects) > 0:
        # Берем произвольный некластеризованный объект
        current_object = objects[random.randint(0, len(objects) - 1)]
        # массив объектов, расположенных на расстоянии <= radius от текущего
        neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < radius and i in objects]
        # находим центр neighbour_objects
        center_object = center_of_objects(dist, neighbour_objects)

        while center_object != current_object:  # пока центр тяжести не стабилизируется
            current_object = center_object
            neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < radius and i in objects]
            center_object = center_of_objects(dist, neighbour_objects)

        # удаляем указанные объекты из выборки (мы их уже кластеризовали)
        for i in neighbour_objects:
            objects.remove(i)

        centers.append(center_object)

        # элементам списка clusters, соответствующим объектам из neighbour_objects
        # присваеваем номер текущего класстера
        for i in neighbour_objects:
            clusters[i] = cur_num
        cur_num += 1

    return clusters, centers


def skat(dist, radius):
    """
    алгоритм СКАТ
    Сначала Вычисляются результаты таксономии  с помощью алгоритма FOREL при радиусе сферы, равном radius.
    Далее процедуры таксономии повторяются с таким же радиусом сфер, но теперь в качестве начальных точек
    выбираются центры, полученные ранее, и формирование каждого нового таксона делается с участием всех  точек.
    В результате обнаруживаются неустойчивые таксоны, которые скатываются к таксонам-предшественникам.
    Решение выдается в виде перечня устойчивых таксонов и указания тех неустойчивых, которые к ним тяготеют.
    :param dist: матрица расстояний между объектами
    :param radius: радиус поиска локальных сгущений
    """

    perv_clusters, centers = forel_for_skat(dist, radius)
    objects = [i for i in range(dist.shape[0])]
    clusters = []
    for current_object in centers:
        # массив объектов, расположенных на расстоянии <= radius от текущего
        neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < radius and i in objects]
        # находим центр neighbour_objects
        center_object = center_of_objects(dist, neighbour_objects)
        while center_object != current_object:  # пока центр тяжести не стабилизируется
            current_object = center_object
            neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < radius and i in objects]
            center_object = center_of_objects(dist, neighbour_objects)
        # элементам списка fcluster, соответствующим объектам из neighbour_objects
        # присваеваем номер текущего класстера
        clusters.append(neighbour_objects)
    return perv_clusters, clusters


def F(centers, clusters, dist):
    """
    Сумма внутриклассовых расстояний до центра. Мера качества разбиения. Стремимся уменьшить.
    """
    return sum(
        sum(dist[c][o[0]] for o in filter(lambda x: x[1] == j+1, enumerate(clusters)))
        for j, c in enumerate(centers))
