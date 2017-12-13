#!/usr/bin/env python3

import tensorflow as tf
import os
import sys
import numpy as np
from utils import _start_shell
from structures import Graph, Node, Edge
from random import shuffle
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
    with tf.device('/cpu:0'):
        x_num_nodes = tf.placeholder(tf.int32)
        y_num_nodes = tf.placeholder(tf.int32)
        x_neighbors = tf.placeholder(tf.int32)
        y_neighbors = tf.placeholder(tf.int32)

        # Static parameters which are shared by each cfg 
        W1 = tf.get_variable("W1", [ACFG_node_dim, emb_size], initializer=tf.random_normal_initializer()) # d x p
        W2 = tf.get_variable("W2", [emb_size, emb_size], initializer=tf.random_normal_initializer()) # p x p
        P_n = []
        for idx in range(relu_layer_num):
            P_n.append(tf.get_variable("P_n_{}".format(idx), [emb_size, emb_size], initializer=tf.random_normal_initializer()))

        graphs_embs = []
        # Build graphs for each cfg
        for idx, graph in enumerate(graphs):
            print("Current graph idx: {}".format(idx))
            if idx == 50:
                break
            # Dynamic parameters for each cfg
            u_v = tf.Variable(tf.zeros([len(graph.nodes), emb_size]), name="u_v_{}".format(idx))

            for t in range(T):
                u_v_t_list = [None] * len(graph.nodes)
                for v in graph.nodes:
                    neighbor_v = list(v.neighbors)
                    if len(neighbor_v) > 0:
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
                    else:
                        u_v_t_list[v.node_id] = tf.nn.embedding_lookup(u_v, v.node_id)
                u_v = tf.reshape(tf.stack(u_v_t_list), [-1, emb_size])
            graph_emb = tf.matmul(W2, tf.reshape(tf.reduce_sum(u_v, 0), [-1, emb_size]), transpose_b=True)
            graphs_embs.append(graph_emb)
    return tf.reshape(tf.stack(graphs_embs), [-1, emb_size])

def build_all_graph_list(dataset):
    all_graph = []
    for sample in dataset['sample']:
        for g in sample:
            if g not in all_graph:
                all_graph.append(g)
    return all_graph


def shuffle_data(dataset):
    data_size = len(dataset['sample'])
    idx_list = [i for i in range(data_size)]
    shuffle(idx_list)
    return np.asarray(dataset['sample'])[idx_list], np.asarray(dataset['label'])[idx_list]


def main(argv):
    if len(argv) != 2:
        print('Usage:\n\tgraph_emb.py <train pickle data>')
        sys.exit(-1)

    with open(sys.argv[1], 'rb') as f:
        learning_data = pickle.load(f)

    cur_epoch = 0
    num_epoch = 1000

    while cur_epoch < num_epoch:
        samples, labels = shuffle_data(learning_data['train'])

        all_graphs = build_all_graph_list(learning_data['train'])

        with tf.variable_scope("siamese") as scope:
            x_graph_id = tf.placeholder(tf.int32)
            y_graph_id = tf.placeholder(tf.int32)
            label = tf.placeholder(tf.float32)
            graphs_embs = graph_emb(all_graphs)

            x = tf.nn.embedding_lookup(graphs_embs, [x_graph_id])
            y = tf.nn.embedding_lookup(graphs_embs, [y_graph_id])

            x = tf.transpose(x)
            y = tf.transpose(y)

            norm_x = tf.nn.l2_normalize(x, 0)
            norm_y = tf.nn.l2_normalize(y, 0)
            cos_similarity = tf.reduce_sum(tf.multiply(norm_x, norm_y))
            print(cos_similarity)
            loss_op = tf.square(cos_similarity - label)
            train_op = tf.train.GradientDescentOptimizer(0.03).minimize(loss_op)
            init_op = tf.global_variables_initializer()

        cur_epoch = 0
        epoch_loss = 0
        num_graph = 100
        real_loss = epoch_loss / len(samples)

        with tf.Session() as sess:
            sess.run(init_op)
            
            for i in range(50):
                loss_sum = 0
                cur_step = 0
                for sample, label_ in zip(samples, labels):
                    x_idx = all_graphs.index(sample[0])
                    y_idx = all_graphs.index(sample[1])
                    if not (x_idx < num_graph and y_idx < num_graph):
                        continue
                    _, loss = sess.run([train_op, loss_op], {x_graph_id: x_idx, y_graph_id: y_idx, label: label_})
                    sys.stdout.write('Epoch: {:10}, Loss: {:15.10f}, Step: {:10}     \r'.format(cur_epoch, epoch_loss, cur_step))
                    sys.stdout.flush()
                    cur_step += 1 
                    loss_sum += loss
                epoch_loss += (loss_sum / len(samples))
                cur_epoch += 1

            _start_shell(locals())



    f_win.close()
    f_linux.close()
    f_arm.close()

if __name__ == '__main__':
    main(sys.argv)

