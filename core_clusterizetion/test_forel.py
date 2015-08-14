from search_query.ya_query import YaQuery
import search_query.serps as wrs
from scipy.spatial.distance import pdist,squareform
import core_clusterizetion.forel

import core_clusterizetion.graph_metods as gr


region = 2
semcorefile = 'C:\\_Work\\vostok\\to_clust.txt'

num_res = 10
queries = [YaQuery(q, region) for q in wrs.queries_from_file(semcorefile)]
print('done')
vectors = [query.get_url_ids(num_res) for query in queries]

print('done')
metrics = lambda u, v: 1 - sum([int(i in v) for i in u]) / num_res
y = pdist(vectors, metrics)

com = gr.shotest_open_path(y, 0.8)
for c in com:
    print([queries[i].query for i in c])



