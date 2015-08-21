"""

Алгоритмы кластеризации ФОРЭЛ и СКАТ. О данных методах можно почитать, например, в
Загоруйко Н. Г. Прикладные методы анализа данных и знаний.

"""

import random
import numpy as np
import core_clusterizetion.core_cluster as cc


def center_of_objects(pdist, neighbour_objects):
    """
    Bозвращает центр тяжести neighbour_objects
    В метрическом пространстве — объект, сумма расстояний до которого минимальна, среди всех внутри сферы

    Parameters
    ----------
        pdist: ndarray, квадратная матрица растояний между всеми объектами задачи
        neighbour_objects:  список номеров объектов для которых нужно найти объект, являющейся центром тяжести

    Returns
    -------
        int, номер объекта, который является центром
    """

    sum_d = np.array([sum(pdist[i][j] for j in neighbour_objects) for i in neighbour_objects])
    return neighbour_objects[sum_d.argmin()]


def n_times_forel(dist, radius, n_tries=200):
    """
    n_tries  раз запускается алгоритм ФОРЭЛ (FOREL)
    На выходе результат попытки с наименьшей суммой внутриклассовых дисперсий.

    Parameters
    ----------
        n_tries: int, число  раз запуска алгоритма.
        dist: ndarray, матрица расстояний между объектами, может быть как в квадратном, так и в конденсированном
             (condensed) виде
        radius: flost, радиус поиска локальных сгущений

    Returns
    -------
        f_cluster: list of int, fclusters[i] - номер кластера, к которому отнесен i-й объект.
        quality: float, сумма внутриклассовых дисперсий для разбиения f_cluster.
        f_centers: list of int, список центров кластеров. centers[j] - номер объекта,
                    являющегося центром кластера с номером j.
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

    Parameters
    ----------
    n_tries: int,
        число  раз запуска алгоритма.
    dist: ndarray,
        матрица расстояний между объектами, может быть как в квадратном, так и в конденсированном виде
    radius: flost,
        радиус поиска локальных сгущений

    Returns
    -------
    clusters: list of lists of int,
        список кластеров, здесь под кластером подразумевается список номеров объектов, которые лежат в кластеры.
    centers: list of int,
        список центров кластеров в том же порядке, что и кластеры в clusters
    """
    f_cluster, quality, centers = n_times_forel(dist, radius, n_tries)

    clusters = [[]] * int(max(f_cluster))
    for i_q, k in enumerate(f_cluster):
        clusters[int(k)-1].append(i_q)

    return clusters, centers


def forel(dist, radius):
    """
    Алгоритм ФОРЭЛ (FOREL).
    На каждом шаге случайным образом выбираем объект из выборки, раздуваем вокруг него сферу радиуса R, внутри этой
    сферы выбираем центр тяжести и делаем его центром новой сферы. То есть мы на каждом шаге двигаем сферу в сторону
    локального сгущения объектов выборки, то есть стараемся захватить как можно больше объектов выборки сферой
    фиксированного радиуса. После того как центр сферы стабилизируется, все объекты внутри сферы с этим центром
    помечаем как кластеризованные и выкидываем их из выборки. Этот процесс мы повторяем до тех пор, пока вся выборка
    не будет кластеризована.

    Parameters
    ----------
    dist: ndarray,
        квадратная матрица расстояний между объектами,
    radius:  numeric
        радиус поиска локальных сгущений (для метрики, корторую я использую 0 < radius < 1)
    Returns
    -------
    clusters: list of int,
        clusters[i] - номер кластера, к которому отнесен i-й объект.
    centers: list of int,
        список центров кластеров. centers[j] - номер объекта, являющегося центром кластера с номером j.
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

    Алгоритм СКАТ
    Сначала Вычисляются результаты таксономии  с помощью алгоритма FOREL при радиусе сферы, равном radius.
    Далее процедуры таксономии повторяются с таким же радиусом сфер, но теперь в качестве начальных точек
    выбираются центры, полученные ранее, и формирование каждого нового таксона делается с участием всех  точек.
    В результате обнаруживаются неустойчивые таксоны, которые скатываются к таксонам-предшественникам.

    Parameters
    ----------
    dist: ndarray,
        матрица расстояний между объектами, может быть как в квадратном, так и в конденсированном виде
    radius: float,
        радиус поиска локальных сгущений

    Returns
    -------
    perv_clusters: list of lists of int,
        Результат кластеризации алгоритмом ФОРЭЛ. Список кластеров, здесь под кластером подразумевается список номеров
        объектов, которые лежат в кластеры.

    clusters: list of lists of int,
        список новых кластеров, которые могут пересекаться, полученых по процедуре, описаной выше.
    """

    perv_clusters, centers = forel_for_skat(dist, radius)
    objects = [i for i in range(dist.shape[0])]
    clusters = []
    for current_object in centers:
        neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < radius and i in objects]
        center_object = center_of_objects(dist, neighbour_objects)
        while center_object != current_object:
            current_object = center_object
            neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < radius and i in objects]
            center_object = center_of_objects(dist, neighbour_objects)
        clusters.append(neighbour_objects)
    return perv_clusters, clusters


def F(centers, clusters, dist):
    """

    Сумма внутриклассовых расстояний до центра. Мера качества разбиения. Стремимся уменьшить.

    Parameters
    ----------
    clusters: list of lists of int,
        список кластеров, здесь под кластером имеется ввиду список номеров объектов, которые лежат в кластеры.
    centers: list of int,
        список центров кластеров в том же порядке, что и кластеры в clusters
    dist: ndarray,
        матрица расстояний между объектами, может быть как в квадратном, так и в конденсированном виде

    Returns
    -------
    F: fload,
        сумма внутриклассовых расстояний до центра
    """
    return sum(
        sum(dist[c][o[0]] for o in filter(lambda x: x[1] == j+1, enumerate(clusters)))
        for j, c in enumerate(centers))
