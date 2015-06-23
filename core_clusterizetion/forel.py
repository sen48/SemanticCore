import random
from scipy.spatial.distance import squareform
import numpy as np

__author__ = 'lvova'





class Forel:

    def __init__(self, pdist, R = 0.9 ):
        # R - ширина поиска локальных сгущений - входной параметр алгоритма

        if len(pdist.shape) == 1:
            self.pdist = squareform(pdist)
        else:
            self.pdist = pdist
        N = pdist.shape[0]
        self.objects = [i for i in range(N)]
        self.R = R # ширина поиска локальных сгущений - входной параметр алгоритма

    def clusterisation_not_finished(self, num_iter=5):
        """все ли объекты кластеризованы"""

        return len(self.objects)>0

    def get_random_object(self):
        """ возвращает произвольный некластеризованный объект """
        return self.objects[random.randint(0, len(self.objects)-1)]


    def get_neighbour_objects(self, current_object):
        """ возвращает массив объектов, расположенных на расстоянии <= R от текущего """
        res = []
        for i, d in enumerate(self.pdist[current_object]):
            if d < self.R and i in self.objects:
                res.append(i)
        return res


    def center_of_objects(self, neighbour_objects):
        """ возвращает центр тяжести указанных объектов
        В метрическом пространстве — объект, сумма расстояний до которого минимальна, среди всех внутри сферы
        """

        sum_d = np.array([sum(self.pdist[i][j] for j in neighbour_objects if j in self.objects)
                          for i in neighbour_objects if i in self.objects])
        return neighbour_objects[sum_d.argmin()]

    def delete_objects(self, neighbour_objects):
        """ удаляет указанные объекты из выборки (мы их уже кластеризовали) """
        [self.objects.remove(neighbour) for neighbour in neighbour_objects]

    def clust(self):
        """
        pdist - матрица расстояний
        R - ширина поиска локальных сгущений - входной параметр алгоритма
        """
        res = list()
        while(self.clusterisation_not_finished()):

           current_object = self.get_random_object()
           neighbour_objects = self.get_neighbour_objects(current_object)
           center_object = self.center_of_objects(neighbour_objects)

           while (center_object != current_object):  #пока центр тяжести не стабилизируется
              current_object = center_object
              neighbour_objects = self.get_neighbour_objects(current_object)
              center_object = self.center_of_objects(neighbour_objects)
           res.append(neighbour_objects)
           self.delete_objects(neighbour_objects)
        return res

