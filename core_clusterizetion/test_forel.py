from scipy.spatial.distance import pdist,squareform
import core_clusterizetion.forel as forel
import core_clusterizetion.graph_metods as gr
from search_query.ya_query import queries_from_file
import core_clusterizetion.core_cluster as core_cluster

import numpy as np


region = 213
semcorefile = 'C:\\_Work\\vostok\\to_clust.txt'

num_res = 10
queries = queries_from_file(semcorefile, region)
vectors = core_cluster.get_queries_vectors(queries, num_res, False)

print('done')
metrics = lambda u, v: 1 - sum([int(i in v) for i in u]) / num_res
y = pdist(vectors, metrics)

com = gr.shotest_open_path(y, 0.8)
for c in com:
    print([queries[i].query for i in c])


clusters = forel.forel_for_skat(y, 0.9)[0]

M = len(clusters)
cdist = np.zeros((M, M))

for i, cl in enumerate(clusters):
    for j in range(i+1, len(clusters)):
        cdist[i][j] = core_cluster.cluster_dist(cl, clusters[j], y)
        cdist[j][i] = cdist[i][j]


clusters_of_cluster_nums = gr.shotest_open_path(cdist, 0.7)
new_clusters = []
print('===================================================')
for cluster_of_nums in clusters_of_cluster_nums:
    new_clusters.append([])
    for num in cluster_of_nums:
        new_clusters[-1] += clusters[num]
        for i in clusters[num]:
            print(queries[i].query)
        print(' ')
    print('+++++++++++++++++++++++++++++++++')
print('===================================================')
print(M)
print(len(clusters_of_cluster_nums))

