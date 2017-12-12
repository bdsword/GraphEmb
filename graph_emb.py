#!/usr/bin/env python3

import tensorflow as tf
import os
import sys
import numpy as np
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


def graph_emb(graphs):
    # Graph:
    #   nodes: {id, code, neighbors}
    #   edges: {src, dest}
    with tf.device('/gpu:0'):
        x_num_nodes = tf.placeholder(tf.int32)
        y_num_nodes = tf.placeholder(tf.int32)
        x_neighbors = tf.placeholder(tf.int32)
        y_neighbors = tf.placeholder(tf.int32)

        # Static parameters which are shared by each cfg 
        W1 = tf.get_variable("W1", [ACFG_node_dim, emb_size], initializer=tf.random_normal_initializer()) # d x p
        W2 = tf.get_variable("W2", [emb_size, emb_size], initializer=tf.random_normal_initializer()) # p x p
        P_n = []
        for _ in range(relu_layer_num):
            P_n.append(tf.Variable(tf.random_normal([emb_size, emb_size])))


        graphs_embs = []
        # Build graphs for each cfg
        for idx, graph in enumerate(graphs):
            # Dynamic parameters for each cfg
            u_v = tf.Variable(tf.zeros([len(graph.nodes), emb_size]), name="u_v_{}".format(idx))

            for t in range(T):
                u_v_t_list = [None] * len(graph.nodes)
                for v in graph.nodes:
                    neighbor_v = list(v.neighbors)
                    neighbor_u = tf.nn.embedding_lookup(u_v, neighbor_v)
                    l_v = tf.reshape(tf.reduce_sum(neighbor_u, 0), [-1, emb_size])

                    sigma_output = simga_function(l_v, P_n)
                    u_v_t = tf.tanh(
                                tf.add(
                                    tf.matmul(
                                        W1,
                                        tf.stack([v.attributes]), transpose_a=True, transpose_b=True
                                    )
                                    , tf.transpose(sigma_output)
                                )
                            )
                    u_v_t_list[v.node_id] = tf.transpose(u_v_t)
                u_v = tf.reshape(tf.stack(u_v_t_list), [-1, emb_size])
            graph_emb = tf.matmul(W2, tf.reshape(tf.reduce_sum(u_v, 0), [-1, emb_size]), transpose_b=True)
            graphs_embs.append(graph_emb)
    return tf.reshape(tf.stack(graphs_embs), [-1, emb_size])


def main(argv):

    with open(sys.argv[1], 'rb') as f:
        learning_data = pickle.load(f)

    f_win = open(argv[1], 'rb')
    f_linux = open(argv[2], 'rb')
    f_arm = open(argv[3], 'rb')
    
    win_graph = pickle.load(f_win)
    linux_graph = pickle.load(f_linux)
    arm_graph = pickle.load(f_arm)
    
    graphs = [win_graph, linux_graph, arm_graph]

    with tf.variable_scope("siamese") as scope:
        x_graph_id = tf.placeholder(tf.int32)
        y_graph_id = tf.placeholder(tf.int32)
        label = tf.placeholder(tf.int32)
        graphs_embs = graph_emb(graphs)

        x = tf.nn.embedding_lookup(graphs_embs, [x_graph_id])
        y = tf.nn.embedding_lookup(graphs_embs, [y_graph_id])

        x = tf.transpose(x)
        y = tf.transpose(y)

        norm_x = tf.nn.l2_normalize(x, 0)
        norm_y = tf.nn.l2_normalize(y, 0)
        cos_similarity = tf.reduce_sum(tf.multiply(norm_x, norm_y))
        loss_op = tf.square(cos_similarity - label)
        train_op = tf.train.GradientDescentOptimizer(0.03).minimize(loss_op)
        init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for _ in range(1000):
            _, loss, x_val, y_val = sess.run([train_op, loss_op, x, y],
                               {x_graph_id: 0, y_graph_id: 2})
            print('loss: {}'.format(loss))

    _start_shell(locals())

    f_win.close()
    f_linux.close()
    f_arm.close()

if __name__ == '__main__':
    main(sys.argv)

