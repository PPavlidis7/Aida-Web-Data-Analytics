import operator
from random import sample

import networkx as nx
import numpy as np
import scipy
from sklearn.preprocessing import normalize


def salsa_algorithm(graph, max_iter=1000, tolerance=1.0e-7, normalized=True):
    """ We used networkx's Hits algorithm. We changed variables' names and made the appropriate modifications"""
    if len(graph) == 0:
        raise ValueError("Graph has zero nodes")

    adjacency_matrix = nx.to_scipy_sparse_matrix(graph, nodelist=list(graph))  # A from slides
    (number_rows, number_cols) = adjacency_matrix.shape  # should be square
    adjacency_matrix_c = normalize(adjacency_matrix, norm='l1', axis=0)  # Normalize adjacency_matrix (columns)
    adjacency_matrix_r = normalize(adjacency_matrix, norm='l1', axis=1)  # Normalize adjacency_matrix (rows)
    authority_matrix = adjacency_matrix_r.T * adjacency_matrix_c  # authority matrix - ArT * Ac from slides
    a = scipy.ones((number_rows, 1)) / number_rows  # initial guess

    # power iteration on authority matrix
    iteration_index = 0
    while True:
        last_a = a
        a = authority_matrix * a
        a = a / a.max()
        # check convergence, l1 norm
        err = scipy.absolute(a - last_a).sum()
        if err < tolerance:
            break
        if iteration_index > max_iter:
            raise nx.PowerIterationFailedConvergence(max_iter)
        iteration_index += 1
    a = np.asarray(a).flatten()
    h = np.asarray(adjacency_matrix_c * a).flatten()  # h=adjacency_matrix_c*a
    if normalized:
        h = h / h.sum()
        a = a / a.sum()
    hubs = dict(zip(graph, map(float, h)))
    authorities = dict(zip(graph, map(float, a)))
    return hubs, authorities


def generate_barabasi_albert_network():
    number_of_nodes = 10 ** 5
    m = 5
    return nx.generators.random_graphs.barabasi_albert_graph(number_of_nodes, m)


def calculate_average_overlap(graph):
    # implement 2-5 assignment's steps
    max_iterations = 10
    # In order to follow PEP8, we initialize all list but we'll use some of them.
    # The memory cost is to low so we won't have any problem
    overlap_salsa_hits = []
    overlap_salsa_page_rank = []
    overlap_hits_page_rank = []
    overlap_salsa_hubs_salsa_authorities = []
    overlap_salsa_authorities_hits_authorities = []
    overlap_salsa_hubs_hits_hubs = []
    overlap_salsa_hubs_page_rank = []
    overlap_salsa_authorities_page_rank = []
    overlap_salsa_hubs_hits_authorities = []
    overlap_salsa_authorities_hits_hubs = []
    overlap_hits_hubs_page_rank = []
    overlap_hits_authorities_page_rank = []
    overlap_hits_hubs_hits_authorities = []

    for iteration_index in range(max_iterations):  # step 6
        route_set = sample(list(graph.nodes()), 2000)  # step 2
        neighbors = set()
        for node in route_set:
            _tmp = [neighbors.add(neighbor) for neighbor in graph.neighbors(node)]

        __sub_graph = graph.subgraph(route_set).copy()
        # check if graph is connected - step 3
        if not nx.is_connected(__sub_graph):
            longest_connected_sub_graph = max(nx.connected_components(__sub_graph), key=len)
            __sub_graph = graph.subgraph(list(longest_connected_sub_graph))

        # step 4
        salsa_hubs, salsa_authorities = salsa_algorithm(barabasi_graph)
        best_salsa_hubs = dict(sorted(salsa_hubs.items(), key=operator.itemgetter(1), reverse=True)[:20])
        best_salsa_authorities = dict(sorted(salsa_authorities.items(), key=operator.itemgetter(1), reverse=True)[:20])

        hits_hubs, hits_authorities = nx.algorithms.link_analysis.hits_scipy(barabasi_graph, 1000)
        best_hits_hubs = dict(sorted(hits_hubs.items(), key=operator.itemgetter(1), reverse=True)[:20])
        best_hits_authorities = dict(sorted(hits_authorities.items(), key=operator.itemgetter(1), reverse=True)[:20])

        page_rank = nx.algorithms.link_analysis.pagerank_scipy(barabasi_graph, max_iter=1000)
        best_page_rank_values = dict(sorted(page_rank.items(), key=operator.itemgetter(1), reverse=True)[:20])

        # overlaps - step 5
        if not nx.is_directed(graph):
            # hubs == authorities
            overlap_salsa_hits.append(__calculate_overlap(best_salsa_hubs, best_hits_hubs))
            overlap_salsa_page_rank.append(__calculate_overlap(best_salsa_hubs, best_page_rank_values))
            overlap_hits_page_rank.append(__calculate_overlap(best_page_rank_values, best_hits_hubs))
        else:
            overlap_salsa_hubs_salsa_authorities.append(__calculate_overlap(best_salsa_hubs, best_salsa_authorities))
            overlap_salsa_authorities_hits_authorities.append(
                __calculate_overlap(best_salsa_authorities, best_hits_authorities))
            overlap_salsa_hubs_hits_hubs.append(__calculate_overlap(best_salsa_hubs, best_hits_hubs))
            overlap_salsa_hubs_page_rank.append(__calculate_overlap(best_salsa_hubs, best_page_rank_values))
            overlap_salsa_authorities_page_rank.append(
                __calculate_overlap(best_salsa_authorities, best_page_rank_values))
            overlap_salsa_hubs_hits_authorities.append(__calculate_overlap(best_salsa_hubs, best_hits_authorities))
            overlap_salsa_authorities_hits_hubs.append(__calculate_overlap(best_salsa_authorities, best_hits_hubs))
            overlap_hits_hubs_hits_authorities.append(__calculate_overlap(best_hits_hubs, best_hits_authorities))
            overlap_hits_hubs_page_rank.append(__calculate_overlap(best_hits_hubs, best_page_rank_values))
            overlap_hits_authorities_page_rank.append(__calculate_overlap(best_hits_authorities, best_page_rank_values))

    if not nx.is_directed(graph):
        print_average_overlap("Average overlap between Salsa and Hits", overlap_salsa_hits)
        print_average_overlap("Average overlap between Salsa and PageRank", overlap_salsa_page_rank)
        print_average_overlap("Average overlap between Hits and PageRank", overlap_hits_page_rank)

        print("\nSalsa hubs:")
        for node_id in sorted(best_salsa_hubs, key=best_salsa_hubs.get, reverse=True):
            print("%d: %.5f" % (node_id, best_salsa_hubs[node_id]))
        print('--'*10)

        print("\nHits hubs: \n")
        for node_id in sorted(best_hits_hubs, key=best_hits_hubs.get, reverse=True):
            print("%d: %.5f" % (node_id, best_hits_hubs[node_id]))
        print('--'*10)

        print("\nPageRank: \n")
        for node_id in sorted(best_page_rank_values, key=best_page_rank_values.get, reverse=True):
            print("%d: %.5f" % (node_id, best_page_rank_values[node_id]))
        print('--' * 10)
    else:
        print_average_overlap("Average overlap between Salsa hubs and Salsa authorities",
                              overlap_salsa_hubs_salsa_authorities)
        print_average_overlap("Average overlap between Salsa authorities and Hits authorities",
                              overlap_salsa_authorities_hits_authorities)
        print_average_overlap("Average overlap between Salsa hubs and Hits hubs", overlap_salsa_hubs_hits_hubs)
        print_average_overlap("Average overlap between Salsa hubs and PageRank", overlap_salsa_hubs_page_rank)
        print_average_overlap("Average overlap between Salsa authorities and PageRank",
                              overlap_salsa_authorities_page_rank)
        print_average_overlap("Average overlap between Salsa hubs and Hits authorities",
                              overlap_salsa_hubs_hits_authorities)
        print_average_overlap("Average overlap between Salsa authorities and Hits hubs",
                              overlap_salsa_authorities_hits_hubs)
        print_average_overlap("Average overlap between Hits hubs Hits authorities", overlap_hits_hubs_hits_authorities)
        print_average_overlap("Average overlap between Hits hubs and PageRank", overlap_hits_hubs_page_rank)
        print_average_overlap("Average overlap between Hits authorities PageRank", overlap_hits_authorities_page_rank)


def __calculate_overlap(var1, var2):
    iteration_fraction = []
    # get variables' keys sorted by value
    var1_nodes = sorted(var1, key=var1.get, reverse=True)
    var2_nodes = sorted(var2, key=var2.get, reverse=True)
    # len(var1) == len(var2)
    for index in range(len(var1)):
        # sort keys in order to get in every iteration the previous set + 1 new node
        # python dictionaries are not sorted
        number_of_same_nodes = len(set(var1_nodes[:index + 1]).intersection(set(var2_nodes[:index + 1])))
        iteration_fraction.append(number_of_same_nodes / (index + 1))
    return np.round(sum(iteration_fraction) / len(var1), 5)


def print_average_overlap(title, overlap):
    print("-" * 50)
    print("{} {}".format(title, np.mean(overlap), 5))


if __name__ == '__main__':
    barabasi_graph = generate_barabasi_albert_network()
    calculate_average_overlap(barabasi_graph)
