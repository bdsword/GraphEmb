#!/usr/bin/env python3

import tensorflow as tf
import os
import sys
import numpy as np
from utils import _start_shell
from structures import Graph, Node, Edge
import pickle
from random import shuffle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

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


def graph_emb(graph, W1, W2, P_n):
    with tf.device('/gpu:0'):
        graphs_embs = []
        u_v = tf.Variable(tf.zeros([len(graph.nodes), emb_size]), name="u_v")

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
    W1_val = None
    W2_val = None
    P_n_val = None

    while cur_epoch < num_epoch:
        samples, labels = shuffle_data(learning_data['train'])
        cur_step = 0
        epoch_loss = 0
        real_loss = epoch_loss / len(samples)
        for sample, label in zip(samples, labels):
            with tf.device('/gpu:0'):
                if W1_val is None:
                    W1 = tf.Variable(tf.random_normal([ACFG_node_dim, emb_size]), name="W1") # d x p
                else:
                    W1 = tf.Variable(W1_val, name="W1")

                if W2_val is None:
                    W2 = tf.Variable(tf.random_normal([emb_size, emb_size]), name="W2") # p x p
                else:
                    W2 = tf.Variable(W2_val, name="W2")

                P_n = []
                if P_n_val is None:
                    for _ in range(relu_layer_num):
                        P_n.append(tf.Variable(tf.random_normal([emb_size, emb_size])))
                else:
                    for P in P_n_val:
                        P_n.append(tf.Variable(P))



            x = graph_emb(sample[0], W1, W2, P_n)
            y = graph_emb(sample[1], W1, W2, P_n)

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
                num_repeat = 10
                loss_sum = 0
                for _ in range(num_repeat):
                    _, loss = sess.run([train_op, loss_op])
                    loss_sum += loss
                sys.stdout.write('Epoch: {:10}, Loss: {:15.10f}, Step: {:10}     \r'.format(cur_epoch, real_loss, cur_step))
                sys.stdout.flush()
                W1_val, W2_val = sess.run([W1, W2])
                P_n_val = sess.run(P_n)
                epoch_loss += (loss_sum / num_repeat)
            cur_step += 1 
        cur_epoch += 1

    _start_shell(locals())

if __name__ == '__main__':
    main(sys.argv)

