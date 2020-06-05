from math import ceil
from textwrap import wrap

import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from sklearn import metrics
from sklearn.cluster import SpectralClustering


def __purity_score(ground_truth, results, graph):
    # get from ground_truth only those that exist as key in results
    __ground_truth = {node_id: ground_truth[node_id] for node_id in results if node_id in ground_truth}
    y_true = [__ground_truth[node] for node in list(graph.nodes) if node in __ground_truth]
    y_pred = [results[node] for node in list(graph.nodes) if node in results]
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def __calculate_modularity(graph, results):
    # group nodes by community_id
    grouped_members = {}
    for community_id in set(results.values()):
        grouped_members[community_id] = set()

    for member, community_id in results.items():
        grouped_members[community_id].add(member)

    communities = [frozenset(__community) for __community in grouped_members.values()]

    # take the subgraph which contains communities' nodes
    __sub_graph = graph.subgraph(list(results.keys())).copy()
    return nx.algorithms.community.quality.modularity(__sub_graph, communities)


def __calculate_tp_tn_fn_fp(ground_truth, results):
    # calculate tp, tn, fn and fp base on 23th slide from 4th lecture
    # every node that has not placed to a community is counted as fn
    fn = len({node_id for node_id in ground_truth if node_id not in results})
    tp, tn, fp = 0, 0, 0
    # a set to be sure that each pair is checked only once
    already_checked = set()
    for node_id, __community in ground_truth.items():
        if node_id not in results:
            continue
        for second_node_id, second_community in ground_truth.items():
            if second_node_id not in results:
                continue
            if second_node_id != node_id and (node_id, second_node_id) not in already_checked and \
                    (second_node_id, node_id) not in already_checked:
                already_checked.add((node_id, second_node_id))
                if __community != second_community:  # if nodes are not similar
                    if results[node_id] == results[second_node_id]:  # if nodes are assigned to same community
                        fp += 1
                    else:
                        tn += 1
                else:  # if nodes are similar
                    if results[node_id] == results[second_node_id]:  # if nodes are assigned to same community
                        tp += 1
                    else:
                        fn += 1
    return tp, tn, fn, fp


def __print_metrics_results(recall, precision, purity, modularity):
    print("Recall = {}\nPrecision = {} \nPurity = {}\nModularity = {}\n\n"
          .format(recall, precision, purity, modularity))


def louvain(graph):
    return community.best_partition(graph)


def spectral_clustering(graph, k_value):
    adj_mat = nx.to_numpy_matrix(graph)
    sc = SpectralClustering(k_value, affinity='precomputed', n_init=100, assign_labels='discretize')
    sc.fit(adj_mat)
    results = {str(node_id): __community for node_id, __community in enumerate(sc.labels_)}
    return results


def get_clique_percolation_communities_metrics(ground_truth, clique_percolation_communities, graph, pos):
    purity = __purity_score(ground_truth, clique_percolation_communities, graph)
    modularity = __calculate_modularity(graph, clique_percolation_communities)
    tp, tn, fn, fp = __calculate_tp_tn_fn_fp(ground_truth, clique_percolation_communities)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Clique percolation metrics:")
    __print_metrics_results(recall, precision, purity, modularity)
    __visualize_communities(graph, clique_percolation_communities, pos, 'Clique percolation')


def get_louvain_metrics(ground_truth, louvain_communities, graph, pos):
    purity = __purity_score(ground_truth, louvain_communities, graph)
    modularity = __calculate_modularity(graph, louvain_communities)
    tp, tn, fn, fp = __calculate_tp_tn_fn_fp(ground_truth, louvain_communities)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Louvain metrics:")
    __print_metrics_results(recall, precision, purity, modularity)
    __visualize_communities(graph, louvain_communities, pos, 'Louvain')


def get_girvan_newman_metrics(ground_truth, girvan_communities, graph, pos):
    purity = __purity_score(ground_truth, girvan_communities, graph)
    modularity = __calculate_modularity(graph, girvan_communities)
    tp, tn, fn, fp = __calculate_tp_tn_fn_fp(ground_truth, girvan_communities)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Girvan newman metrics:")
    __print_metrics_results(recall, precision, purity, modularity)
    __visualize_communities(graph, girvan_communities, pos, 'Girvan newman')


def get_spectral_clustering_metrics(ground_truth, spectral_clustering_communities, graph, pos):
    purity = __purity_score(ground_truth, spectral_clustering_communities, graph)
    modularity = __calculate_modularity(graph, spectral_clustering_communities)
    tp, tn, fn, fp = __calculate_tp_tn_fn_fp(ground_truth, spectral_clustering_communities)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Spectral clustering metrics:")
    __print_metrics_results(recall, precision, purity, modularity)
    __visualize_communities(graph, spectral_clustering_communities, pos, 'Spectral clustering')


def visualize_ground_truth(graph, ground_truth, pos):
    __visualize_communities(graph, ground_truth, pos, 'Ground Truth')


def __visualize_communities(graph, communities, pos, title):
    # in case communities do not have all nodes
    communities_to_show = {-1: []}
    for node_id in graph.nodes:
        if node_id not in communities:
            communities_to_show[-1].append(node_id)
        else:
            if communities[node_id] not in communities_to_show:
                communities_to_show[communities[node_id]] = []
            communities_to_show[communities[node_id]].append(node_id)

    # generate legend values
    label_legend = {}
    for __community in list(communities_to_show.keys()):
        if __community != -1:
            label_legend[__community] = __community
        else:
            label_legend[__community] = "Without Community"

    plt.axis('off')
    fig = plt.figure(figsize=(15, 15))
    # in order to map string to color
    vmin = min(list(map(int, label_legend.keys())))
    vmax = max(list(map(int, label_legend.keys())))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.rainbow
    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for __community in sorted(label_legend.keys()):
        nx.draw_networkx_nodes(graph, pos, node_size=45, node_color=np.atleast_2d(scalar_map.to_rgba(__community)),
                               nodelist=communities_to_show[__community], label=label_legend[__community])
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    plt.title("\n".join(wrap(title)))

    # place legend at left side of plot in order to not cover part of plot
    plt.legend(ncol=ceil(len(label_legend) / 50), labelspacing=0.05, bbox_to_anchor=(0, 0), loc="lower right",
               fancybox=True, shadow=True)
    plt.show()
