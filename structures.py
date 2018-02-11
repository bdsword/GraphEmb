class Node:
    def __init__(self, node_id, code, neighbors, attributes):
        self.node_id = node_id
        self.code = code 
        self.neighbors = neighbors
        self.attributes = attributes


class DirAwareNode:
    def __init__(self, node_id, code, incomes, outcomes, attributes):
        self.node_id = node_id
        self.code = code
        self.incomes = incomes
        self.outcomes = outcomes
        self.neighbors = list(incomes) + list(outcomes)
        self.attributes = attributes


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
