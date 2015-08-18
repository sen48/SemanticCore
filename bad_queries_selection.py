"""
Тут нет ничего полезного
"""

import statistics
import igraph as ig
import numpy as np
from scipy.spatial.distance import pdist, squareform
from search_engine_tk.ya_query import queries_from_file
from search_engine_tk.serp_metrics import number_of_common_urls


MINUSWORDS = ['где', 'как', 'фото', 'своими руками', 'видео', 'скачать', 'бесплатно', 'икея', 'леруа', 'мерлен',
              'официальный сайт', 'мастер класс', 'подключить', 'подручных', 'сломалась', 'ремонт', 'к чему', 'что',
              'какую', 'ютуб', 'youtube', 'форум', 'сравнить', 'отличия']
cities = []


def remove_queries_with_minus_words(queries, minus_words=None):
    if minus_words:
        minuswords = MINUSWORDS + minus_words
    else:
        minuswords = MINUSWORDS

    p = [query for query in queries if all(word not in query.query for word in minuswords)]
    return p


def get_vecs(url_or_dom, fname, num_results=10, ):
    if url_or_dom not in ['url', 'hostname']:
        raise Exception('ddd')

    vectors = np.zeros((num_points, num_results))
    all_domains = []
    for i, query in enumerate(queries):
        print('{} {}'.format(i, query.query))
        query.get_serp(10)
        if url_or_dom == 'url':
            vec = query.get_url_ids(num_results)
        else:
            vec = []
            domains = query.get_domains(num_results)
            for domain in domains:
                if domain in all_domains:
                    vec.append(all_domains.index(domain))
                else:
                    vec.append(len(all_domains))
                    all_domains.append(domain)
        vectors[i, :] = vec
    np.savetxt(fname, vectors.T)
    print(vectors.shape)
    return vectors


def load_vecs(fname):
    return np.loadtxt(fname, unpack=True)


def get_graph(X, fname=None):
    metric = lambda u, v: number_of_common_urls(u, v) / num_res

    d = squareform(pdist(X, metric=metric))
    M, N = d.shape

    G = ig.Graph(M)
    for i in range(M):
        for j in range(i+1, N):
            if d[i, j] > 0:
                G.add_edge(i, j, weight=d[i, j])
    if fname:
        f = "gml"
        G.save(fname, format=f)
    return G


def load_graph(fname):
    f = "gml"
    return ig.load(fname, format=f)


def remove_queries(G, threshold=0.4):
    rank = [(ind, r) for ind, r in enumerate(G.pagerank(weights='weight', directed=False, niter=10000, eps=0.0001))]
    dist = []
    sor = sorted(rank, key=lambda v: -v[1])

    for j, (ind, r) in enumerate(sor[1:]):
        dist.append(abs(G.degree(ind)-G.degree(sor[j][0])))

    delta = max(dist)-threshold*(max(dist)-min(dist))
    result = []
    for j, (i, r) in enumerate(sor[1:-1]):
        if dist[j] > delta and dist[j+1] > delta:
            result.append((i, statistics.mean([dist[j], dist[j+1]])))
            try:
                G.delete_vertices(i)
            except:
                print(i)
                raise
    return result


if __name__ == '__main__':

    queries = remove_queries_with_minus_words(queries_from_file('c:/_Work/vostok/to_clust.csv', 213))

    num_points = len(queries)

    num_res = 10

    X = get_vecs('hostname', 'vostok1_host', num_res)
    X = load_vecs("vostok1_host")  # "data2"
    print(X.shape)
    G = get_graph(X, "graph1_vostok_host.txt")

    G = load_graph("graph1_vostok_host.txt")
    print('===============================')
    res = remove_queries(G, 0.5)
    for (i_r, d) in sorted(res, key=lambda v: -v[1]):
        print(queries[i_r].query, d)
    print('===============================')

    comp = sorted(G.clusters(mode='STRONG'), key=len, reverse=True)
    S = G
    for c in comp[1:]:
        print([queries[i_c].query for i_c in c])
        S.delete_vertices(c)

    for j_i, i_i in enumerate(S.vs.indices):
        if i_i != j_i:
            print('fsdf!!!', i_i, j_i)

    print(len(S.vs.indices))

    print('===============================')
    res = remove_queries(S, 0.5)
    for (i_r, d) in sorted(res, key=lambda v: -v[1]):
        print(queries[i_r].query, d)

    print('===============================')


    #comp = G.clusters(mode='STRONG')

    '''print()

    for group in sorted(comp, key=len, reverse=True):
        if len(group) > 1:
            print([queries[i].query for i in group])
    print('Коэффициент кластеризации', len(comp))
    comp = G.clusters(mode='WEAK').giant()
    print([queries[idx].query for idx, v in enumerate(comp.vs)])
    print('Коэффициент кластеризации', len(comp))
    for group in sorted(comp, key=len, reverse=True)[0:1]:
        if len(group) > 1:
            print([queries[i].query for i in group])
        gr = G.subgraph(group)
        print('len:{} diam:{}'.format(len(group), nx.diameter(gr)))
        clustering_coefficient = nx.average_clustering(gr)
        print('Коэффициент кластеризации', clustering_coefficient)
        if len(gr.nodes()) > 2 and clustering_coefficient < 0.8:
            for node in gr.nodes():
                w = 0
                for n in G.neighbors(node):
                    if node < n:
                        w += nx.get_edge_attributes(gr, 'weight')[node, n]
                    else:
                        w += nx.get_edge_attributes(gr, 'weight')[n, node]
                if w == 1 or w == 0:
                    print(queries[node].query, w, len(G.neighbors(node)))
                    gr.remove_node(node)
                    G.remove_node(node)
                    #break
            #else:
               # break
        #print(nx.is_connected(gr))
        #print(nx.is_biconnected(gr))
        if len(gr.nodes()) == 0:
            print('Hello')
            continue

        print('len:{} diam:{}'.format(len(gr.nodes()), nx.diameter(gr)))
        clustering_coefficient = nx.average_clustering(gr)
        print('Коэффициент кластеризации', clustering_coefficient)
        print('Центр графа:', [queries[i].query for i in nx.center(gr)])
        print([queries[i].query for i in group])
    core_clusterizetion.visual.plot_graph(G)"""

    """
    G = nx.Graph()
    for i in range(M):
        for j in range(i+1, N):
            if d[i, j] > 0:
                G.add_edge(i, j, weight=d[i, j] > 0)
    print('Коэффициент кластеризации', nx.average_clustering(G))
    comp = sorted(nx.connected_components(G), key=len, reverse=True)
    for group in comp:
        gr = G.subgraph(group)
        print('len:{} diam:{}'.format(len(group), nx.diameter(gr)))
        clustering_coefficient = nx.average_clustering(gr)
        print('Коэффициент кластеризации', clustering_coefficient)
        if len(gr.nodes()) > 2 and clustering_coefficient < 0.8:
            for node in gr.nodes():
                w = 0
                for n in G.neighbors(node):
                    if node < n:
                        w += nx.get_edge_attributes(gr, 'weight')[node, n]
                    else:
                        w += nx.get_edge_attributes(gr, 'weight')[n, node]
                if w == 1 or w == 0:
                    print(queries[node].query, w, len(G.neighbors(node)))
                    gr.remove_node(node)
                    G.remove_node(node)
                    #break
            #else:
               # break
        #print(nx.is_connected(gr))
        #print(nx.is_biconnected(gr))
        if len(gr.nodes()) == 0:
            print('Hello')
            continue

        print('len:{} diam:{}'.format(len(gr.nodes()), nx.diameter(gr)))
        clustering_coefficient = nx.average_clustering(gr)
        print('Коэффициент кластеризации', clustering_coefficient)
        print('Центр графа:', [queries[i].query for i in nx.center(gr)])
        print([queries[i].query for i in group])
    core_clusterizetion.visual.plot_graph(G)
    #print(nx.diameter(G))"""

    """from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(algorithm='randomized', n_components=round(M*0.5), n_iter=5,
                       random_state=42, tol=0.0)
    X = d
    svd.fit(X)

    print(svd.explained_variance_ratio_)
    print(sum(svd.explained_variance_ratio_))
    print(X)
    X = svd.transform(X)
    print(X)

    classifiers = {
    "One-Class SVM": svm.OneClassSVM(nu=0.95 * 0.25 + 0.05,
                                     kernel="rbf", gamma=0.1),
    "robust covariance estimator": EllipticEnvelope(contamination=.1)}



    # Example settings
    n_samples = 277
    outliers_fraction = 0.25
    clusters_separation = [0, 1, 2]

    # define two outlier detection tools to be compared
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                         kernel="rbf", gamma=0.1),
        "robust covariance estimator": EllipticEnvelope(contamination=.1)}

    # Compare given classifiers under given settings


    # Fit the problem with varying cluster separation

    for i, (clf_name, clf) in enumerate(classifiers.items()):
            # fit the data and tag outliers
            clf.fit(X)
            y_pred = clf.decision_function(X).ravel()
            print(y_pred)
            threshold = stats.scoreatpercentile(y_pred,
                                                100 * outliers_fraction)
            y_pred = y_pred > threshold
            qs = [q.query for i, q in enumerate(queries) if y_pred[i]]
            print(len(qs))
            print(qs)
            qs = [q.query for i, q in enumerate(queries) if not y_pred[i]]
            print(len(qs))
            print(qs)
            #n_errors = (y_pred != ground_truth).sum()
            # plot the levels lines and the points"""'''



