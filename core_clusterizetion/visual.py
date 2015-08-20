"""
Отсюда используется только plot_dendrogram
"""

import random

import networkx as nx
import matplotlib.pyplot as plt
import pylab as pl
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from matplotlib import rc


rc('font', **{'family': 'verdana'})
rc('text.latex', unicode=True)
rc('text.latex', preamble='\\usepackage[utf8]{inputenc}')
rc('text.latex', preamble='\\usepackage[russian]{babel}')


def plot_data(data_frame, metric):
    x = data_frame.get_values()
    y = pdist(x, metric)
    y = [r * int(r < 20) for r in y]
    d = squareform(y)
    plot_graph(graph=d)


def dendrogram1(data_frame, metric):
    x = data_frame.T.get_values()
    n_texts = len(data_frame.columns)
    d = (pdist(x, metric))

    # Compute and plot first dendrogram.s
    figure = pl.figure(figsize=(8, 8))
    ax1 = figure.add_axes([0.09, 0.1, 0.2, 0.6])
    y = sch.linkage(d, method='weighted', metric=metric)
    z1 = sch.dendrogram(y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = figure.add_axes([0.3, 0.71, 0.6, 0.2])
    y = sch.linkage(d, method='average', metric=metric)
    z2 = sch.dendrogram(y)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot distance matrix.
    axmatrix = figure.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = z1['leaves']
    idx2 = z2['leaves']
    d = squareform(d)
    d = d[idx1, :]
    d = d[:, idx2]
    im = axmatrix.matshow(d, aspect='auto', origin='lower', cmap=pl.cm.YlGnBu)

    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axmatrix.set_xticks(range(n_texts))
    axmatrix.set_xticklabels([data_frame.columns[t] for t in z2['leaves']], minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    pl.xticks(rotation=-90, fontsize=8)

    axmatrix.set_yticks(range(n_texts))
    axmatrix.set_yticklabels([data_frame.columns[t] for t in z1['leaves']], minor=False)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()

    # Plot colorbar.
    axcolor = figure.add_axes([0.94, 0.1, 0.02, 0.6])
    pl.colorbar(im, cax=axcolor)
    return figure


def plot_dendrogram(z, fcl=None, labels=None, fig=12):

    if len(fcl) < 2:
        llf = None
    else:
        llf = lambda idx: "{} {}".format(labels[idx], fcl[idx])

    def clr(k):
        n = int(k * round(16 ** 6 / (max(fcl) + 1)))
        res = str(hex(n)[2:])
        while len(res) < 6:
            res = '0' + res
        return '#{}'.format(res)

    num_keys = len(fcl)

    def get_num_cl(zk):
        if zk < len(fcl):
            return fcl[zk]
        elif get_num_cl(zk - len(fcl)):
            pass

    def get_cl(row, lr):
        if z[row, lr] < len(fcl):
            return fcl[int(z[row, lr])]
        else:
            a = get_cl(z[row, lr] - len(fcl), 0)
            b = get_cl(z[row, lr] - len(fcl), 1)
            if a == b:
                return a
            else:
                return len(fcl)

    def lcf(k):
        k -= num_keys
        a = get_cl(k, 0)
        if a == len(fcl):
            return 'b'
        b = get_cl(k, 1)
        if a == b:
            return clr(a)  # colors[divmod(get_cl(k, 0), len(colors))[1]]
        else:
            return 'b'

    plt.figure(fig, figsize=(14, 10))
    plt.title(u'Дендрограмма')
    plt.axis('tight')  # 'scaled')
    sch.dendrogram(z, orientation='left', leaf_label_func=llf, link_color_func=lcf)
    # plt.savefig("dendrogram{}.png".format(fig))
    plt.legend()
    plt.grid()
    plt.show()


def plot_matrix(d, keys=None):
    figure = plt.figure(50, figsize=(8, 8))
    axmatrix = figure.add_axes([0.3, 0.1, 0.6, 0.6])
    im = axmatrix.matshow(d, cmap=pl.cm.YlGnBu)
    axmatrix.set_xticks([])
    if keys:
        axmatrix.set_yticks(range(len(keys)))
        axmatrix.set_yticklabels(keys)

    # Plot colorbar.
    axcolor = figure.add_axes([0.94, 0.1, 0.02, 0.6])
    pl.colorbar(im, cax=axcolor)
    plt.show()


def plot_linkage_union_dist(z):
    plt.figure(1)
    plt.clf()
    y = z[:, 2]
    l = len(y)
    x = [l - i for i in range(0, l)]
    plt.plot(x, y, 'ro')
    plt.ylabel('number of clusters')
    plt.ylabel('distance')
    plt.savefig("pl.png")
    plt.show()


def plot_graph(graph):
    plt.close('all')
    plt.figure(1)
    plt.clf()

    if isinstance(graph, nx.Graph):
        nx.draw(graph,
                node_size=40,
                node_color='g',
                vmin=0.0,
                vmax=1.0,
                with_labels=False)

    else:
        nx.draw(nx.from_numpy_matrix(graph),
                node_size=40,
                node_color='g',
                vmin=0.0,
                vmax=1.0,
                with_labels=False)

    plt.savefig("atlas.png", dpi=75)
    plt.show()


def plot_clustered_graph(dist_matrix, labels=None):
    plt.close('all')
    plt.figure(1)
    plt.clf()
    n_clusters = max(labels) + 1
    print('n_clusters = {}'.format(n_clusters))
    g_g = nx.Graph()
    for k in range(0, n_clusters):
        class_members = labels == k
        class_dist = dist_matrix[class_members].T[class_members]
        g = nx.from_numpy_matrix(class_dist)
        g_g = nx.disjoint_union(g_g, g)

    # color nodes the same in each connected subgraph
    for g in nx.connected_component_subgraphs(g_g):
        c = [random.random()] * nx.number_of_nodes(g)  # random color...
        nx.draw(g,

                node_size=40,
                node_color=c,
                vmin=0.0,
                vmax=1.0,
                with_labels=False)
    plt.savefig("atlas.png", dpi=75)
    plt.show()



