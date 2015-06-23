from search_query.ya_query import YaQuery
import search_query.serps as wrs
from scipy.spatial.distance import pdist,squareform
import core_clusterizetion.forel
import numpy as np

__author__ = 'lvova'
region = 213
semcorefile = 'C:\\_Work\\lightstar\\to_filter.csv'

num_res = 10
queries = [YaQuery(q, region) for q in wrs.queries_from_file(semcorefile)]
vectors = [query.get_url_ids(num_res) for query in queries]

print('done')
metrics = lambda u, v: 1 - sum([int(i in v) for i in u]) / num_res
y = squareform(pdist(vectors, metrics))
N, M = y.shape

F = core_clusterizetion.forel.Forel(y, R=0.8)

for l in sorted(F.clust(), key=len, reverse=True):
    print(len(l))
    print([queries[i].query for i in l])
"""for i in range(N):
    neighbour = core_clusterizetion.forel.get_neighbour_objects(i,y)
    print(neighbour)
    print(i)
    print(core_clusterizetion.forel.center_of_objects(neighbour,y))"""


