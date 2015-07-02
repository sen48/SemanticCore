from search_query.ya_query import YaQuery
import search_query.serps as wrs
from scipy.spatial.distance import pdist,squareform
import core_clusterizetion.forel

import core_clusterizetion.graph_metods as gr


'''region = 2
semcorefile = 'C:\\_Work\\lightstar\\to_clust.txt'

num_res = 10
queries = [YaQuery(q, region) for q in wrs.queries_from_file(semcorefile)]
print('done')
vectors = [query.get_url_ids(num_res) for query in queries]

print('done')
metrics = lambda u, v: 1 - sum([int(i in v) for i in u]) / num_res
y = pdist(vectors, metrics)

com = gr.shotest_open_path(y, 0.8)
for c in com:
    print([queries[i].query for i in c])'''


def kir():
    import numpy as np
    C1 = np.float64(10)
    C2 = np.float64(100)
    N = 500
    M = 500
    k = np.float64(1/M)
    h = np.float64(1/N)
    D = np.float64(5)
    r = D *k / h**2
    print(r)
    S = 1 - 2*r
    C = np.ones((M+1, N+1))
    for j in range(M+1):
        C[j][0] = C1
        C[j][N] = C2

    for i in range(N+1):
        C[0][i] = [p for p in np.arange(C1, C2, (C2-C1)/N)]

    for j in range(1, M + 1):
        for i in range(1, N):
            C[j][i] = r*C[j-1][i-1] + S*C[j-1][i] + r*C[j-1][i+1]
        print(C[j][0:3], C[j][-3:])

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    X = np.array([[k*j for i in range(0, N+1)] for j in range(0, M+1)])
    Y = np.array([[h*i for i in range(0, N+1)] for j in range(0, M+1)])
    Z = C
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

    plt.figure()
    plt.plot([h*i for i in range(0, N+1)], C[M][:])

    plt.show()



kir()


