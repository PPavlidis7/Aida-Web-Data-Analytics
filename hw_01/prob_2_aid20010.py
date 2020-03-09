import networkx as nx


def read_data():
    with open('CA-HepTh.txt') as f:
        authors_data = f.read()

    authors_data = [tuple(pair.split()) for pair in authors_data.split('\n')[4:-1]]

    # create networks
    authors_network = nx.Graph(authors_data)
    return nx.number_of_nodes(authors_network), nx.number_of_edges(authors_network)


def generate_erdos_renyi_network(number_of_nodes):
    G = nx.generators.random_graphs.fast_gnp_random_graph(number_of_nodes, 0.00054)
    # TODO: statistics


def generate_network_with_power_law_distribution(number_of_nodes):
    G = nx.generators.random_graphs.powerlaw_cluster_graph(number_of_nodes, 3, 0.0005)
    # TODO: statistics


def generate_wattz_strogatz_network(number_of_nodes):
    G = nx.generators.random_graphs.watts_strogatz_graph(number_of_nodes, 6, 0.0005)
    # TODO: statistics


def generate_barabasi_albert_network(number_of_nodes):
    G = nx.generators.random_graphs.barabasi_albert_graph(number_of_nodes, 3)
    # TODO: statistics


def main():
    number_of_nodes, number_of_edges = read_data()
    # generate_erdos_renyi_network(number_of_nodes)
    generate_network_with_power_law_distribution(number_of_nodes)
    # generate_wattz_strogatz_network(number_of_nodes)
    generate_barabasi_albert_network(number_of_nodes)


if __name__ == '__main__':
    main()