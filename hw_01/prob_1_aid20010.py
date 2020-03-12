import operator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def read_data():
    with open('CA-HepTh.txt') as f:
        authors_data = f.read()
    with open('Wiki-Vote.txt') as f:
        votes_data = f.read()

    # preprocess data
    authors_data = [tuple(pair.split()) for pair in authors_data.split('\n')[4:-1]]
    votes_data = [tuple(pair.split()) for pair in votes_data.split('\n')[4:-1]]

    # create networks
    authors_network = nx.Graph(authors_data)
    votes_network = nx.DiGraph(votes_data)
    return authors_data, votes_data, authors_network, votes_network


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


def plot_degrees(given_network, directed=False):
    if not directed:
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
        __plot_helper_log(x, density, 'Degree distribution for undirected graph in log scale')

        # Get 20 logarithmically spaced bins between kmin and kmax
        bin_edges = np.linspace(kmin, kmax, num=10)
        # histogram the data into these bins
        density, _ = np.histogram(degrees, bins=bin_edges, density=True)
        log_be = np.log10(bin_edges)
        x = 10**((log_be[1:] + log_be[:-1])/2)
        __plot_helper_linear(x, density, 'Degree distribution for undirected graph in linear scale')
    else:
        degrees_to_plot = {
            'Degree distribution for directed graph ': [given_network.degree(node) for node in given_network],
            'In-degree distribution for directed graph ': [given_network.in_degree(node) for node in given_network],
            'Out-degree distribution for directed graph ': [given_network.out_degree(node) for node in given_network]
        }
        for title, degrees in degrees_to_plot.items():
            kmin = min(degrees)
            kmax = max(degrees)

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
            __plot_helper_log(x, density, title + 'in log scale')

            # Get 20 logarithmically spaced bins between kmin and kmax
            bin_edges = np.linspace(kmin, kmax, num=10)
            # histogram the data into these bins
            density, _ = np.histogram(degrees, bins=bin_edges, density=True)
            log_be = np.log10(bin_edges)
            x = 10 ** ((log_be[1:] + log_be[:-1]) / 2)
            __plot_helper_linear(x, density, title + 'in linear scale')


def get_network_values(given_network, directed=False):
    print('Number of distinct nodes: ', nx.number_of_nodes(given_network))
    print('Number of nodes with a self-loop: ', nx.number_of_selfloops(given_network))
    if directed:
        __undirected_graph = given_network.to_undirected()
        print('Number of undirected edges in the network: ', nx.number_of_edges(__undirected_graph))
        print('Number of directed edges in the network: :', nx.number_of_edges(given_network))
        print('Number of of reciprocated edges: :', (nx.algorithms.reciprocity(given_network) *
                                                     len(list(given_network.edges))) / 2)
        print('Number of sink nodes: ',
              len([node for node, out_degree in given_network.out_degree() if out_degree == 0]))
        print('Number of source nodes: ',
              len([node for node, in_degree in given_network.in_degree() if in_degree == 0]))
        given_network_in_degrees = dict(given_network.in_degree)
        given_network_out_degrees = dict(given_network.out_degree)
        print(
            'The min and max out-degree: %d and %d' % (
                min(given_network_in_degrees.items(), key=operator.itemgetter(1))[1],
                max(given_network_in_degrees.items(), key=operator.itemgetter(1))[1]))
        print(
            'The min and max in-degree: %d and %d' % (
                min(given_network_out_degrees.items(), key=operator.itemgetter(1))[1],
                max(given_network_out_degrees.items(), key=operator.itemgetter(1))[1]))
        print('The average degree, in-degree, out-degree: %f , %f and %f' % (
            sum(dict(given_network.degree).values()) / len(dict(given_network.degree)),
            sum(given_network_in_degrees.values()) / len(given_network_in_degrees),
            sum(given_network_out_degrees.values()) / len(given_network_out_degrees)
        ))
        plot_degrees(given_network, True)
        try:
            print('Network diameter: ', nx.algorithms.distance_measures.diameter(given_network))
        except nx.exception.NetworkXError:
            print('Found infinite path length because the digraph is not strongly connected')
            print('I will find diameter for the largest connected subgraph')
            longest_connected_subgraph = max(nx.strongly_connected_components(given_network), key=len)
            new_smaller_graph = given_network.subgraph(list(longest_connected_subgraph))
            print('Network diameter: ', nx.algorithms.distance_measures.diameter(new_smaller_graph))
    else:
        __directed_graph = given_network.to_directed()
        print('Number of undirected edges in the network: ', nx.number_of_edges(given_network))
        print('Number of directed edges in the network: :', nx.number_of_edges(__directed_graph))
        # in a undirected graph both (u,v) and (v,u) edges exist
        print('Number of of  edges: :', nx.number_of_edges(given_network))
        print('The average degree: %f' % (sum(dict(given_network.degree).values()) / len(dict(given_network.degree))))
        plot_degrees(given_network)
        print('Network diameter: ', nx.algorithms.distance_measures.diameter(given_network))
        try:
            print('Network diameter: ', nx.algorithms.distance_measures.diameter(given_network))
        except nx.exception.NetworkXError:
            print('Found infinite path length because the digraph is not strongly connected')
            print('I will find diameter for the largest connected subgraph')
            longest_connected_subgraph = max(nx.connected_components(given_network), key=len)
            new_smaller_graph = given_network.subgraph(list(longest_connected_subgraph))
            print('Network diameter: ', nx.algorithms.distance_measures.diameter(new_smaller_graph))

    print(
        'The min and max node degree: %d and %d' % (
            min(dict(given_network.degree).items(), key=operator.itemgetter(1))[1],
            max(dict(given_network.degree).items(), key=operator.itemgetter(1))[1]))


def main():
    authors, votes, authors_network, votes_network = read_data()
    print('Authors network: \n')
    get_network_values(authors_network)
    print('-'*20)
    print('Votes network: \n')
    get_network_values(votes_network, True)


if __name__ == '__main__':
    main()
