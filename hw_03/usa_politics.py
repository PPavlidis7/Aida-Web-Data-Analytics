import networkx as nx

import shared_functions as sf


def read_data():
    data = nx.Graph(nx.read_gml('./data/polblogs.gml'))
    return data


def clique_percolation(graph):
    k_value = 8
    com = list(nx.algorithms.community.k_clique_communities(graph, k_value))
    # write results to a file
    data = ''
    for item in com:
        data += "(%s)\n" % str(set(item))

    with open('results/clique_percol_blogs.txt', 'w') as f:
        f.write(data)


def girvan_newman(graph):
    email_girvan_newman = list(nx.algorithms.community.centrality.girvan_newman(graph))

    # write results to a file
    data = ''
    for item in email_girvan_newman:
        data += "%s\n" % str(item)

    with open('results/girvan_blogs.txt.txt', 'w') as f:
        f.write(data)


# Functions for data and results preprocessing
def get_clique_percolation_results(ground_truth):
    # group ground truth data by community_id
    # initialize
    grouped_members = {}
    for community_id in set(ground_truth.values()):
        grouped_members[community_id] = set()

    for member, community_id in ground_truth.items():
        grouped_members[community_id].add(member)

    with open('results/clique_percol_blogs.txt') as f:
        data = [line.split() for line in f.readlines()]

    # preprocess data and store them in a dictionary in format node_id: community_id
    results = {}
    duplicates = {}
    for community_id, __community in enumerate(data):
        for member in eval("".join(__community)):
            if member in results:
                if member in duplicates:
                    duplicates[member] += [community_id]
                else:
                    duplicates[member] = [community_id]
            else:
                results[member] = community_id

    # handle duplicates
    # for each node that has been placed to multiple communities, find best community based on ground truth and select
    # this as node's community. To evaluate communities we do
    # (number of nodes in this communities that are also in the same ground truth)/ length of community.
    # The community with biggest value is selected.
    for member, in_communities in duplicates.items():
        first_membership = results[member]
        membership_communities = {}
        for __community_id in in_communities + [first_membership]:
            membership_communities[__community_id] = eval("".join(data[__community_id]))

        community_evaluations = {}
        for __community_id, community_members in membership_communities.items():
            community_evaluations[__community_id] = \
                len(community_members.intersection(grouped_members[ground_truth[member]])) / len(community_members)

        results[member] = max(community_evaluations, key=community_evaluations.get)

    return results


def get_girvan_newman_results():
    with open('results/girvan_blogs.txt') as f:
        data = [line.split() for line in f.readlines()]

    # preprocess data and store them in a dictionary in format node_id: community_id
    results = {}
    community_id = 0
    for communities in data:
        # when we find at least two communities with length greater or equal with 5
        if len([__com for __com in eval("".join(communities)) if len(__com) > 5]) >= 2:
            for __community in eval("".join(communities)):
                for member in __community:
                    results[member] = community_id
                community_id += 1
            break

    return results


if __name__ == '__main__':
    graph = read_data()
    ground_truth = nx.get_node_attributes(graph, 'value')

    # compute graph layout in order all nodes in all graphs to have the same position
    pos = nx.spring_layout(graph)

    # clique_percolation(graph)
    louvain_communities = sf.louvain(graph)
    # girvan_newman(graph)
    spectral_clustering_communities = sf.spectral_clustering(graph, 2)

    # modify keys. Algorithm returns node_ids ar 0,1,2...
    # We need to replace them with their label in order to continue
    fixed_spectral_clustering_communities = {}
    for node_id, node_label in enumerate(list(graph.nodes)):
        fixed_spectral_clustering_communities[node_label] = spectral_clustering_communities[str(node_id)]

    clique_percolation_communities = get_clique_percolation_results(ground_truth)
    girvan_communities = get_girvan_newman_results()

    # metrics' results
    sf.get_clique_percolation_communities_metrics(ground_truth, clique_percolation_communities, graph, pos)
    sf.get_louvain_metrics(ground_truth, louvain_communities, graph, pos)
    sf.get_girvan_newman_metrics(ground_truth, girvan_communities, graph, pos)
    sf.get_spectral_clustering_metrics(ground_truth, fixed_spectral_clustering_communities, graph, pos)

    # plot ground truth
    sf.visualize_ground_truth(graph, ground_truth, pos)
