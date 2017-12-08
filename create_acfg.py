#!/usr/bin/env python3
import pydotplus
import sys
from structures import Graph, Node, Edge
from attributes import attributes_funcs
import pickle
import numpy as np
from utils import _start_shell


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
    if len(argv) != 4:
        print('Usage:\n\tcreate_acfg.py dot_file_path arch output_file_path')
        sys.exit(-1)

    graph = create_acfg_from_file(sys.argv[1], sys.argv[2])
    with open(sys.argv[3], 'wb') as f:
        pickle.dump(graph, f)


if __name__ == '__main__':
    main(sys.argv)
