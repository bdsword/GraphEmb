#!/usr/bin/env python3

import tensorflow as tf
import os
import sys
from utils import _start_shell
from structures import Graph, Node, Edge
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

T = 5 
relu_layer_num = 3
emb_size = 64
ACFG_node_dim = 4

def simga_function(input_l_v, P_n):
    output = input_l_v
    for idx in range(relu_layer_num):
        output = tf.matmul(P_n[idx], tf.nn.relu(output), transpose_b=True) # [p, p] x [p, 1]= [p, 1]
        output = tf.transpose(output) # [1, p]
    return output


def graph_emb(graph):
    # Graph:
    #   nodes: {id, code, neighbors}
    #   edges: {src, dest}
    with tf.device('/gpu:0'):
        u_v = tf.Variable(tf.zeros([len(graph.nodes), emb_size]), name="u_v")
        
        W1 = tf.get_variable("W1", [ACFG_node_dim, emb_size], initializer=tf.random_normal_initializer()) # d x p
        W2 = tf.get_variable("W2", [emb_size, emb_size], initializer=tf.random_normal_initializer()) # p x p
        P_n = []
        for _ in range(relu_layer_num):
            P_n.append(tf.Variable(tf.random_normal([emb_size, emb_size])))

        for t in range(T):
            u_v_t_list = [None] * len(graph.nodes)
            for v in graph.nodes:
                neighbor_v = list(v.neighbors)
                neighbor_u = tf.nn.embedding_lookup(u_v, neighbor_v)
                l_v = tf.reshape(tf.reduce_sum(neighbor_u, 0), [-1, emb_size])

                simga_output = simga_function(l_v, P_n)
                u_v_t = tf.tanh(
                            tf.add(
                                tf.matmul(
                                    W1,
                                    tf.stack([v.attributes]), transpose_a=True, transpose_b=True
                                )
                                , simga_output
                            )
                        )

                u_v_t_list[v.node_id] = tf.transpose(u_v_t)
            u_v = tf.stack(u_v_t_list)
    return tf.matmul(W2, tf.reduce_sum(u_v, 0))


def main(argv):
    with open(argv[1], 'rb') as f:
        graph = pickle.load(f)
        with tf.variable_scope("siamese") as scope:
            x = graph_emb(graph)
            scope.reuse_variables()
            y = graph_emb(graph)
        _start_shell(locals())

if __name__ == '__main__':
    main(sys.argv)
