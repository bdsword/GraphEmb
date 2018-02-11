#!/usr/bin/env python3
import pydotplus
import sys
from structures import Graph, Node, DirAwareNode
from attributes import attributes_funcs
import pickle
import numpy as np
from utils import _start_shell
import re
import os
import glob


Debug = True

def get_attributes(code, arch):
    attrs = []
    for func in attributes_funcs:
        attr = func(code, arch)
        attrs.append(attr)
    x = np.array(attrs)
    x = x.astype(np.float32)
    return x


def find_entry_node(graph):
    '''
    We treat the node which has no income nodes and most child nodes as entry node.
    '''
    max_outcome = -1
    entry_node = -1
    for node in graph.nodes:
        child_node_count = get_child_node_number(graph, node)
        if len(node.incomes) == 0 and child_node_count > max_outcome:
            max_outcome = child_node_count
            entry_node = node
    return entry_node


def get_child_node_number(graph, node):
    reactable_nodes = []
    queue = [node]
    child_node_count = 0
    while len(queue) > 0:
        cur_node = queue.pop()
        if cur_node not in reactable_nodes:
            reactable_nodes.append(cur_node)
            child_node_count += 1
            for outcome_node_id in cur_node.outcomes:
                queue.append(graph.nodes[outcome_node_id])
    return child_node_count - 1


def construct_graph_begin_with_entry_node(graph, entry_node):
    reactable_nodes = []
    queue = [entry_node]
    while len(queue) > 0:
        cur_node = queue.pop()
        if cur_node not in reactable_nodes:
            reactable_nodes.append(cur_node)

            for outcome_node_id in cur_node.outcomes:
                queue.append(graph.nodes[outcome_node_id])

    update_table = {}
    for idx, node in enumerate(reactable_nodes):
        update_table[node.node_id] = idx

    node_list = []
    for node_id, node in enumerate(reactable_nodes):
        incomes = []
        outcomes = []
        for income in node.incomes:
            incomes.append(update_table[income])
        for outcome in node.outcomes:
            outcomes.append(update_table[outcome])

        neighbor_id_list = incomes + outcomes
        if Debug:
            node_list.append(DirAwareNode(node_id, node.code, incomes, outcomes, node.attributes))
        else:
            node_list.append(Node(node_id, node.code, neighbor_id_list, node.attributes))

    graph.nodes = node_list

    return graph


def create_acfg_from_file(filename):
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
            node = DirAwareNode(counter, code, set(), set(), None)
            nodes[node_id] = node
            counter += 1

    for dot_edge in dot_edges:
        src_id = dot_edge.get_source()
        dst_id = dot_edge.get_destination()
        nodes[src_id].outcomes.add(nodes[dst_id].node_id)
        nodes[dst_id].incomes.add(nodes[src_id].node_id)

    nodes_list = [None] * len(nodes)
    for node_name in nodes:
        nodes_list[nodes[node_name].node_id] = nodes[node_name]

    graph = Graph(nodes_list)
    return graph


def get_link_number(graph):
    if not Debug:
        raise Exception('This function can only be called in debug mode.')
    count = 0
    for node in graph.nodes:
        count += len(node.incomes)
    return count


def main(argv):
    if len(argv) != 3:
        print('Usage:\n\t{} <input binrary list> <output path>'.format(argv[0]))
        sys.exit(-1)

    binary_graph_dict = dict()
    with open(argv[1], 'r') as f:
        lines = f.readlines()
        files = [line.strip('\n') for line in lines if len(line) != 0]

    for f in files:
        if not os.path.isfile(f):
            print('{} is not a reqular file.'.format(f))
            sys.exit(-2)
        dot_f = f + '.dot'
        if not os.path.isfile(dot_f):
            print('No dot file for {}.'.format(f))
            sys.exit(-3)

        print('>> Processing {}'.format(f))
        graph = create_acfg_from_file(dot_f)
        entry_node = find_entry_node(graph)
        dense_graph = construct_graph_begin_with_entry_node(graph, entry_node)
        binary_graph_dict[f] = dense_graph

    output_path = argv[2]
    with open(output_path, 'wb') as f:
        pickle.dump(binary_graph_dict, f)


if __name__ == '__main__':
    main(sys.argv)
