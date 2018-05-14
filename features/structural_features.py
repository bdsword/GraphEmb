import networkx as nx


def betweenness_centrality(graph):
    betweenness_centrality = nx.algorithms.centrality.betweenness_centrality(graph)
    for centrality in betweenness_centrality:
        betweenness_centrality[centrality] *= 30.66
    return betweenness_centrality

def num_offspring(graph):
    num_offspring = {}
    for node_id in graph.nodes:
        num_offspring[node_id] = len(nx.algorithms.dag.descendants(graph, node_id)) * 198.67
    return num_offspring

structural_features = [betweenness_centrality, num_offspring]

