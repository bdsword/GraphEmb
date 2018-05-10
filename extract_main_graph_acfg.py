#!/usr/bin/env python3
import networkx as nx
import sys
import pickle
from utils import _start_shell
import os
import glob
import argparse
import numpy as np


def extract_main_graph(file_path):
    graph = nx.drawing.nx_pydot.read_dot(file_path)
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


def add_attributes_to_graph(graph, function_embs, default_dims):
    """
    file_path: Path to the dot file
    function_embs: A dict map function name to function embedding
    """
    for node in graph.nodes:
        func_name = graph.nodes[node]['label'].lstrip('"').rstrip('\\l"')
        if func_name in function_embs:
            graph.nodes[node]['attributes'] = function_embs[func_name]
        else:
            graph.nodes[node]['attributes'] = np.zeros(default_dims)

    return graph


def check_function_embs_and_get_dims(function_embs):
    first_dim = None
    for func_name in function_embs:
        if first_dim == None:
            first_dim = np.shape(function_embs[func_name])
        else:
            if first_dim != np.shape(function_embs[func_name]):
                raise ValueError('The dimensions of each vectors in FunctionEmbsPlk should be all the same.')
    return first_dim


def main(argv):
    parser = argparse.ArgumentParser(description='Extracts subgraph from a dot file which contains most number of child nodes.')
    parser.add_argument('CallGraph', help='A dot file contains the call graph.')
    parser.add_argument('FunctionEmbsPlk', help='A pickle file contains a list of function embeddings in the given call graph.')
    parser.add_argument('OutputPickleFile', help='Path to write output pickle file contains the ACG of input call graph.')
    parser.add_argument('--debug', metavar='EnableDebugMode', type=bool, default=False, help='Enable debug mode. (Default: False)')
    parser.add_argument('--verbose', metavar='EnableVerboseMode', type=bool, default=False, help='Enable debug mode. (Default: False)')
    args = parser.parse_args()


    function_embs = None
    with open(args.FunctionEmbsPlk, 'rb') as f:
        function_embs = pickle.load(f)
    default_dims = check_function_embs_and_get_dims(function_embs)

    if not os.path.isfile(args.CallGraph):
        print('{} does not exist.'.format(args.CallGraph))
        sys.exit(-1)
    if os.path.splitext(args.CallGraph)[1] != '.dot':
        print('{} is not a dot file.'.format(args.CallGraph))
        sys.exit(-2)

    if args.verbose:
        print('>> Processing {}'.format(args.CallGraph))
    main_graph = extract_main_graph(args.CallGraph)
    attributed_main_graph = add_attributes_to_graph(main_graph, function_embs, default_dims)

    output_path = args.OutputPickleFile
    with open(output_path, 'wb') as f:
        pickle.dump(attributed_main_graph, f)


if __name__ == '__main__':
    main(sys.argv)
