import pickle
import os
import networkx as nx


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

