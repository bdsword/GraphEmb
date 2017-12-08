class Node:
    def __init__(self, node_id, code, neighbors, attributes):
        self.node_id = node_id
        self.code = code 
        self.neighbors = neighbors
        self.attributes = attributes


class Edge:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
