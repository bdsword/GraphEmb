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


def add_attributes_to_graph(graph, function_embs):
    """
    file_path: Path to the dot file
    function_embs: A dict map function name to function embedding
    """
    for node in graph.nodes:
        func_name = graph.nodes[node]['label']
        graph.nodes[node]['attributes'] = function_embs[func_name]
    return graph


def main(argv):
    parser = argparse.ArgumentParser(description='Extracts subgraph from a dot file which contains most number of child nodes.')
    parser.add_argument('ListFile', help='A text file contains a list of binary file path to be processed.')
    parser.add_argument('FunctionEmbsPlk', help='A pickle file contains a list of function embeddings.')
    parser.add_argument('OutputPickleFile', help='Path to write output pickle file.')
    parser.add_argument('--debug', metavar='EnableDebugMode', type=bool, default=False, help='Enable debug mode. (Default: False)')
    args = parser.parse_args()

    binary_graph_dict = dict()
    with open(args.ListFile, 'r') as f:
        lines = f.readlines()
        files = [line.strip('\n') for line in lines if len(line.strip('\n')) != 0]

    function_embs = None
    with open(args.FunctionEmbsPlk, 'rb') as f:
        function_embs = pickle.load(f)

    for f in files:
        if not os.path.isfile(f):
            print('{} is not a reqular file.'.format(f))
            sys.exit(-2)
        dot_f = f + '.dot'
        if not os.path.isfile(dot_f):
            print('No dot file for {}.'.format(f))
            sys.exit(-3)

        print('>> Processing {}'.format(f))
        main_graph = extract_main_graph(dot_f)
        attributed_main_graph = add_attributes_to_graph(main_graph, function_embs)
        binary_graph_dict[f] = attributed_main_graph

    output_path = args.OutputPickleFile
    with open(output_path, 'wb') as f:
        pickle.dump(binary_graph_dict, f)


if __name__ == '__main__':
    main(sys.argv)
