from search_query.ya_query import queries_from_file
from search_query.serp_metrics import number_of_common_urls

import core_clusterizetion.visual
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA

import networkx as nx

if __name__ == '__main__':


    print("getting queries")
    queries = queries_from_file('c:/_Work/spetc1.txt', 213)

    num_points = len(queries)
    num_res = 10
    counter = 0
    '''for query in queries:
        counter += 1
        print('{} {}'.format(counter, query.query))
        query.get_serp(10)'''
    print("getting serps")
    """vectors = np.zeros((num_points, num_res))
    counter = 0
    for i, query in enumerate(queries):
        counter += 1
        print('{} {}'.format(counter, query.query))
        vectors[i, :] = query.get_url_ids(num_res)
    np.savetxt("data", vectors.T)
    print(vectors.shape)"""
    vectors = np.loadtxt("data", unpack=True)
    print(vectors.shape)


    X = vectors
    metric = lambda u, v: number_of_common_urls(u, v) / num_res
    d = squareform(pdist(X, metric=metric))
    print(d.shape)
    M, N = d.shape
    G = nx.Graph()
    for i in range(M):
        for j in range(i+1, N):
            if d[i, j] > 0:
                G.add_edge(i, j, weight=(d[i, j] > 0))
    print('Коэффициент кластеризации', nx.average_clustering(G))
    for group in sorted(nx.connected_components(G), key=len, reverse=True):
        gr = G.subgraph(group)
        print('len:{} diam:{}'.format(len(group), nx.diameter(gr)))
        clustering_coefficient = nx.average_clustering(gr)
        print('Коэффициент кластеризации', clustering_coefficient)
        if len(gr.nodes()) > 2 and clustering_coefficient < 0.8:
            for node in gr.nodes():
                if nx.degree(gr)[node] < 3:
                    print(queries[node].query)
                    gr.remove_node(node)
                    G.remove_node(node)
                    #break
            #else:
               # break
        print(nx.is_connected(gr))
        print(nx.is_biconnected(gr))
        print('len:{} diam:{}'.format(len(gr.nodes()), nx.diameter(gr)))
        clustering_coefficient = nx.average_clustering(gr)
        print('Коэффициент кластеризации', clustering_coefficient)
        print('Центр графа:', [queries[i].query for i in nx.center(gr)])
        print([queries[i].query for i in group])
    core_clusterizetion.visual.plot_graph(G)
    #print(nx.diameter(G))

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
            # plot the levels lines and the points"""



