import operator

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def read_data():
    with open('CA-HepTh.txt') as f:
        authors_data = f.read()
    authors_data = [tuple(pair.split()) for pair in authors_data.split('\n')[4:-1]]
    # create networks
    authors_network = nx.Graph(authors_data)
    return nx.number_of_nodes(authors_network)


def __plot_helper_log(x, density, title):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, density, marker='o', linestyle='none')
    plt.xlabel(r"degree $k$", fontsize=16)
    plt.ylabel(r"$P(k)$", fontsize=16)
    plt.title(title)
    # remove right and top boundaries because they're ugly
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Show the plot
    plt.show()


def __plot_helper_linear(x, density, title):
    fig = plt.figure(figsize=(6, 4))
    plt.loglog(x, density, marker='o', linestyle='none')
    plt.xlabel(r"degree $k$", fontsize=16)
    plt.ylabel(r"$P(k)$", fontsize=16)
    plt.title(title)
    # remove right and top boundaries because they're ugly
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Show the plot
    plt.show()


def plot_degrees(given_network):
    degrees = [given_network.degree(node) for node in given_network]
    kmin = min(degrees)
    kmax = max(degrees)

    # in order to fix RuntimeWarning: divide by zero encountered in log10
    if kmax == 0:
        kmax = 0.000000001
    if kmin == 0:
        kmin = 0.000000001

    # create plot log scale
    # Get 10 logarithmically spaced bins between kmin and kmax
    bin_edges = np.logspace(np.log10(kmin), np.log10(kmax), num=10)
    # histogram the data into these bins
    density, _ = np.histogram(degrees, bins=bin_edges, density=True)
    # "x" should be midpoint (IN LOG SPACE) of each bin
    log_be = np.log10(bin_edges)
    x = 10 ** ((log_be[1:] + log_be[:-1]) / 2)
    __plot_helper_log(x, density, 'Degree distribution for graph in log scale')

    # Get 20 logarithmically spaced bins between kmin and kmax
    bin_edges = np.linspace(kmin, kmax, num=10)
    # histogram the data into these bins
    density, _ = np.histogram(degrees, bins=bin_edges, density=True)
    log_be = np.log10(bin_edges)
    x = 10 ** ((log_be[1:] + log_be[:-1]) / 2)
    __plot_helper_linear(x, density, 'Degree distribution for graph in linear scale')


def generate_erdos_renyi_network(number_of_nodes):
    __graph = nx.generators.random_graphs.fast_gnp_random_graph(number_of_nodes, 0.00054)
    get_network_characteristics(__graph, 'Erdős-Rényi network')


def generate_network_with_power_law_distribution(number_of_nodes):
    __graph = nx.generators.random_graphs.powerlaw_cluster_graph(number_of_nodes, 3, 0.0005)
    get_network_characteristics(__graph, 'Network with power-law distribution')


def generate_wattz_strogatz_network(number_of_nodes):
    __graph = nx.generators.random_graphs.watts_strogatz_graph(number_of_nodes, 6, 0.0005)
    get_network_characteristics(__graph, 'Network using the Wattz Strogatz model')


def generate_barabasi_albert_network(number_of_nodes):
    __graph = nx.generators.random_graphs.barabasi_albert_graph(number_of_nodes, 3)
    get_network_characteristics(__graph, 'Network using the Barabasi-Albert model')


def get_network_characteristics(given_network, title):
    print(title)
    print('Number of distinct nodes: ', nx.number_of_nodes(given_network))
    print('Number of nodes with a self-loop: ', nx.number_of_selfloops(given_network))
    print('Number of undirected edges in the network: ', nx.number_of_edges(given_network))
    print(
        'The min and max node degree: %d and %d' % (
            min(dict(given_network.degree).items(), key=operator.itemgetter(1))[1],
            max(dict(given_network.degree).items(), key=operator.itemgetter(1))[1]))
    print('The average degree: %f' % (sum(dict(given_network.degree).values()) / len(dict(given_network.degree))))
    try:
        print('Network diameter: ', nx.algorithms.distance_measures.diameter(given_network))
    except nx.exception.NetworkXError:
        print('Found infinite path length because the digraph is not strongly connected')
        print('I will find diameter for the largest connected subgraph')
        longest_connected_subgraph = max(nx.connected_components(given_network), key=len)
        new_smaller_graph = given_network.subgraph(list(longest_connected_subgraph))
        print('Network diameter: ', nx.algorithms.distance_measures.diameter(new_smaller_graph))
    plot_degrees(given_network)
    print('\n')


def main():
    number_of_nodes = read_data()
    generate_erdos_renyi_network(number_of_nodes)
    generate_network_with_power_law_distribution(number_of_nodes)
    generate_wattz_strogatz_network(number_of_nodes)
    generate_barabasi_albert_network(number_of_nodes)


if __name__ == '__main__':
    main()
