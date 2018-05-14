import pickle
import os
import networkx as nx
from features.statistical_features import statistical_features
from features.structural_features import structural_features


def extract_main_graph(file_path):
    graph = read_graph(file_path)
    undired_graph = graph.to_undirected()
    sub_graphs = nx.connected_component_subgraphs(undired_graph)
    max_node_num = -1
    main_sub_graph = None
    for i, sg in enumerate(sub_graphs):
        num_nodes = sg.number_of_nodes()
        if num_nodes > max_node_num:
            max_node_num = num_nodes
            main_sub_graph = sg
    return main_sub_graph


def read_graph(path):
    cached_plk = path + '.nxgraph.plk'
    if not os.path.isfile(cached_plk):
        graph = nx.drawing.nx_pydot.read_dot(path)
        with open(cached_plk, 'wb') as f:
            pickle.dump(graph, f)
    else:
        with open(cached_plk, 'rb') as f:
            graph = pickle.load(f)
    return graph


def get_statistical_features(graph, arch):
    features = dict()
    for feature_func in statistical_features:
        features[feature_func.__name__] = dict()
        for node in graph.nodes:
            code = graph.nodes[node]['label']
            features[feature_func.__name__][node] = float(feature_func(code, arch))
    return features


def get_structural_features(graph):
    features = dict()
    for feature_func in structural_features:
        features[feature_func.__name__] = feature_func(graph)
    return features


def create_acfg_from_file(file_path, arch):
    graph = read_graph(file_path)
    features_dict = get_statistical_features(graph, arch)
    for feature_name in features_dict:
        nx.set_node_attributes(graph, features_dict[feature_name], feature_name)

    feature_dict = get_structural_features(graph)
    for feature_name in feature_dict:
        nx.set_node_attributes(graph, feature_dict[feature_name], feature_name)

    return graph

