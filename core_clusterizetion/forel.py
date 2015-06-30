import random
import numpy as np
from scipy.spatial.distance import squareform


_norm_pdist = lambda pdist: squareform(pdist) if pdist.ndim == 1 else pdist


def center_of_objects(pdist, neighbour_objects):
    """ возвращает центр тяжести neighbour_objects
        В метрическом пространстве — объект, сумма расстояний до которого минимальна, среди всех внутри сферы
        """
    sum_d = np.array([sum(pdist[i][j] for j in neighbour_objects) for i in neighbour_objects])
    return neighbour_objects[sum_d.argmin()]


def forel(dist, R):
    dist = _norm_pdist(dist)
    f_cluster = np.ones(dist.shape[0])
    quality = None
    for i in range(200):
        clusters, centers = forel_step(dist, R)
        q = F(centers, clusters, dist)
        if not quality or q < quality:
            quality = q
            f_cluster = clusters
    return f_cluster, quality


def forel_step(dist, R):
    objects = [i for i in range(dist.shape[0])]
    cur_num = 1
    clusters = np.ones(dist.shape[0])
    centers = []
    while len(objects) > 0:
        # Берем произвольный некластеризованный объект
        current_object = objects[random.randint(0, len(objects) - 1)]
        # массив объектов, расположенных на расстоянии <= R от текущего
        neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < R and i in objects]
        # находим центр neighbour_objects
        center_object = center_of_objects(dist, neighbour_objects)

        while center_object != current_object:  # пока центр тяжести не стабилизируется
            current_object = center_object
            neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < R and i in objects]
            center_object = center_of_objects(dist, neighbour_objects)

        # удаляем указанные объекты из выборки (мы их уже кластеризовали)
        for i in neighbour_objects: objects.remove(i)

        centers.append(center_object)

        # элементам списка fcluster, соответствующим объектам из neighbour_objects
        # присваеваем номер текущего класстера
        for i in neighbour_objects:
            clusters[i] = cur_num
        cur_num += 1

    return clusters, centers


def forel_for_skat(dist, R):
    dist = _norm_pdist(dist)
    f_cluster = np.ones(dist.shape[0])
    quality = None
    centers = []
    for i in range(200):
        fclusters, cent = forel_step(dist, R)
        q = F(cent, fclusters, dist)
        if not quality or q < quality:
            quality = q
            f_cluster = fclusters
            centers = cent

    clusters = [[] for i in range(int(max(f_cluster)))]
    for i, k in enumerate(f_cluster):
        clusters[int(k)-1].append(i)

    return clusters, centers


def skat(dist, R):
    perv_clusters, centers = forel_for_skat(dist, R)
    objects = [i for i in range(dist.shape[0])]
    clusters = []
    for current_object in centers:
        # массив объектов, расположенных на расстоянии <= R от текущего
        neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < R and i in objects]
        # находим центр neighbour_objects
        center_object = center_of_objects(dist, neighbour_objects)
        while center_object != current_object:  # пока центр тяжести не стабилизируется
            current_object = center_object
            neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < R and i in objects]
            center_object = center_of_objects(dist, neighbour_objects)
        # элементам списка fcluster, соответствующим объектам из neighbour_objects
        # присваеваем номер текущего класстера
        clusters.append(neighbour_objects)
    return perv_clusters, clusters


def cluster(dist, R, old_centers = None):
    dist = _norm_pdist(dist)
    f_cluster = np.ones(dist.shape[0])
    centers = []
    cur_num = 1
    objects = [i for i in range(dist.shape[0])]

    if not old_centers:
            old_centers = objects

    while len(objects) > 0:
        # Берем произвольный некластеризованный объект из списка возможнух центров

        current_object = old_centers[random.randint(0, len(old_centers) - 1)]

        # массив объектов, расположенных на расстоянии <= R от текущего
        neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < R and i in objects]
        # находим центр neighbour_objects
        center_object = center_of_objects(dist, neighbour_objects)

        while center_object != current_object:  # пока центр тяжести не стабилизируется
            current_object = center_object
            neighbour_objects = [i for i, d in enumerate(dist[current_object]) if d < R and i in objects]
            center_object = center_of_objects(dist, neighbour_objects)

        # удаляем указанные объекты из выборки (мы их уже кластеризовали)
        for i in neighbour_objects:
            objects.remove(i)
            if i in old_centers:
                old_centers.remove(i)
        if len(old_centers) == 0:
            old_centers = objects
        # элементам списка fcluster, соответствующим объектам из neighbour_objects
        # присваеваем номер текущего класстера
        centers.append(center_object)
        for i in neighbour_objects:
            f_cluster[i] = cur_num
        cur_num += 1

    return f_cluster, F(centers, f_cluster, dist), centers


def F(centers, clusters, dist):
    return sum(
        sum(dist[c][o[0]] for o in filter(lambda x: x[1] == j+1, enumerate(clusters)))
        for j, c in enumerate(centers))
