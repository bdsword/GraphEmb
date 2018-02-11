#!/usr/bin/env python3
import pydotplus
import sys
from structures import Graph, Node
from attributes import attributes_funcs
import pickle
import numpy as np
from utils import _start_shell
import re
import os
import glob


def get_attributes(code, arch):
    attrs = []
    for func in attributes_funcs:
        attr = func(code, arch)
        attrs.append(attr)
    x = np.array(attrs)
    x = x.astype(np.float32)
    return x


def create_acfg_from_file(filename, arch):
    dot_graph = pydotplus.parser.parse_dot_data(open(filename, 'r').read())
    dot_nodes = dot_graph.get_nodes()
    dot_edges = dot_graph.get_edges()
    edges = []
    nodes = dict()
    counter = 0
    for dot_node in dot_nodes:
        node_id = dot_node.get_name()
        code = dot_node.get_label()
        if code:
            node = Node(counter, code, set(), get_attributes(code, arch))
            nodes[node_id] = node
            counter += 1

    for dot_edge in dot_edges:
        src_id = dot_edge.get_source()
        dst_id = dot_edge.get_destination()
        nodes[src_id].neighbors.add(nodes[dst_id].node_id)
        nodes[dst_id].neighbors.add(nodes[src_id].node_id)
        edges.append(Edge(nodes[src_id].node_id, nodes[dst_id].node_id))

    nodes_list = [None] * len(nodes)
    for node_name in nodes:
        nodes_list[nodes[node_name].node_id] = nodes[node_name]

    graph = Graph(nodes_list, edges)
    return graph


def main(argv):
    if len(argv) != 3:
#         print('Usage:\n\tcreate_acfg.py dot_file_path arch output_file_path')
        print('Usage:\n\tcreate_acfg.py <src folder> <output pickle>')
        sys.exit(-1)

    src_folder = argv[1]
    if os.path.isdir(src_folder) == False:
        print('The target path is not a folder.')
        sys.exit(-1)

    arch_func_graph = dict()
    for file_path in glob.iglob(os.path.join(src_folder, '**', '*.dot'), recursive=True):
        file_name = os.path.basename(file_path)
        pattern = r'(.+)~(.+)\.dot'
        items = re.findall(pattern, file_name)[0]
        arch = 'x86'
        bin_name = items[0]
        function_name = items[1]
        try:
            graph = create_acfg_from_file(file_path, arch)
        except:
            os.remove(file_path)
            continue
        if arch not in arch_func_graph:
            arch_func_graph[arch] = dict()
        if function_name not in arch_func_graph[arch]:
            arch_func_graph[arch][function_name] = dict()
        arch_func_graph[arch][function_name][bin_name] = graph

    with open(sys.argv[2], 'wb') as f:
        pickle.dump(arch_func_graph, f)


if __name__ == '__main__':
    main(sys.argv)

