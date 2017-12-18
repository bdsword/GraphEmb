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
MAX_NEIGHBORS_NUM = 50
relu_layer_num = 2
emb_size = 64
attributes_dim = 4 # d

def simga_function(input_l_v, P_n):
    # N: number of nodes
    output = input_l_v
    for idx in range(relu_layer_num):
        output = tf.matmul(P_n[idx], tf.nn.relu(output), transpose_b=True) # [p, p] x [p, N]= [p, N]
        output = tf.transpose(output) # [N, p]
    return output

def build_emb_graph(neighbors, attributes, u_init):
    with tf.device('/gpu:0'):
        # Static parameters which are shared by each cfg 
        W1 = tf.get_variable("W1", [attributes_dim, emb_size], initializer=tf.random_normal_initializer(stddev=0.1)) # d x p
        W2 = tf.get_variable("W2", [emb_size, emb_size], initializer=tf.random_normal_initializer(stddev=0.1)) # p x p
        P_n = []
        for idx in range(relu_layer_num):
            P_n.append(tf.get_variable("P_n_{}".format(idx), [emb_size, emb_size], initializer=tf.random_normal_initializer(stddev=0.1)))

        # Dynamic parameters for each cfg
        u_v = u_init

        for t in range(T):
            neighbors_u = tf.nn.embedding_lookup(u_v, neighbors)
            l_vs = tf.reduce_sum(neighbors_u, 1)
            sigma_output = simga_function(l_vs, P_n) # [N, p]
            u_v_transposed = tf.tanh(
                        tf.add(
                            tf.matmul(
                                W1,
                                attributes, transpose_a=True, transpose_b=True
                            ) # [p, d] x [d, N] = [p, N]
                            , tf.transpose(sigma_output) # [p, N]
                        )
                    ) # [p, N]
            u_v = tf.transpose(u_v_transposed) # [N, p]
        graph_emb = tf.transpose(tf.matmul(W2, tf.reshape(tf.reduce_sum(u_v[1:], 0), [-1, emb_size]), transpose_b=True)) # ([p, p] x [p, 1])^T = [p, 1]^T = [1, p]
    return graph_emb


def shuffle_data(dataset):
    data_size = len(dataset['sample'])
    idx_list = [i for i in range(data_size)]
    shuffle(idx_list)
    return np.asarray(dataset['sample'])[idx_list], np.asarray(dataset['label'])[idx_list]


def get_graph_info_mat(graph):
    neighbors = []
    attributes = []

    neighbors.append(np.zeros(MAX_NEIGHBORS_NUM))
    attributes.append(np.zeros(attributes_dim))
    for node in graph.nodes:
        if MAX_NEIGHBORS_NUM < len(node.neighbors):
            raise ValueError('Number of neightbors is larger than MAX_NEIGHBORS_NUM: {} > MAX_NEIGHBORS_NUM'.format(len(node.neighbors)))
        ns = np.pad(list(node.neighbors), (0, MAX_NEIGHBORS_NUM - len(node.neighbors)), 'constant', constant_values=0)
        neighbors.append(ns)
        attributes.append(node.attributes)

    return neighbors, attributes, np.zeros((len(graph.nodes), emb_size))

def main(argv):
    if len(argv) != 5:
        print('Usage:\n\tgraph_emb.py <train pickle data> <saver folder> <load model> <start ipython>')
        sys.exit(-1)

    if not os.path.isdir(argv[2]):
        print('Saver folder should be a valid folder.')
        sys.exit(-2)
    saver_path = argv[2]
    load_model = int(argv[3])
    start_ipython = int(argv[4])

    with open(argv[1], 'rb') as f:
        learning_data = pickle.load(f)

    if start_ipython == 0:
        with tf.variable_scope("siamese") as scope:
            # Build Training Graph
            neighbors_left = tf.placeholder(tf.int32, shape=(None, MAX_NEIGHBORS_NUM)) # N x MAX_NEIGHBORS_NUM
            attributes_left = tf.placeholder(tf.float32, shape=(None, attributes_dim)) # N x d
            u_init_left = tf.placeholder(tf.float32, shape=(None, emb_size)) # N x p
            graph_emb_left = build_emb_graph(neighbors_left, attributes_left, u_init_left) # N x p

            scope.reuse_variables()

            neighbors_right = tf.placeholder(tf.int32, shape=(None, MAX_NEIGHBORS_NUM)) # N x MAX_NEIGHBORS_NUM
            attributes_right = tf.placeholder(tf.float32, shape=(None, attributes_dim)) # N x d
            u_init_right = tf.placeholder(tf.float32, shape=(None, emb_size)) # N x p
            graph_emb_right = build_emb_graph(neighbors_right, attributes_right, u_init_right) # N x p

            label = tf.placeholder(tf.float32)

            norm_emb_left = tf.nn.l2_normalize(graph_emb_left, 1)
            norm_emb_right = tf.nn.l2_normalize(graph_emb_right, 1)
            cos_similarity = tf.reduce_sum(tf.multiply(norm_emb_left, norm_emb_right))
            loss_op = tf.square(cos_similarity - label)

        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss_op)
    else: 
        with tf.variable_scope("siamese") as scope:
            # Bulid Inference Graph
            neighbors_test = tf.placeholder(tf.int32, shape=(None, MAX_NEIGHBORS_NUM)) # N x MAX_NEIGHBORS_NUM
            attributes_test = tf.placeholder(tf.float32, shape=(None, attributes_dim)) # N x d
            u_init_test = tf.placeholder(tf.float32, shape=(None, emb_size)) # N x p
            graph_emb = build_emb_graph(neighbors_test, attributes_test, u_init_test) # N x p

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        
        if load_model == 1:
            states = tf.train.get_checkpoint_state(saver_path)
            saver.restore(sess, states.model_checkpoint_path)
        if start_ipython == 1:
            _start_shell(locals(), globals())
        else:
            num_epoch = 50
            epoch_loss = -1
            total_step = 0
            for cur_epoch in range(num_epoch):
                loss_sum = 0
                cur_step = 0
                correct = 0
                samples, labels = learning_data['test']['sample'], learning_data['test']['label']
                if cur_epoch != 0:
                    for sample, ground_truth in zip(samples, labels):
                        neighbors_l, attributes_l, u_init_l = get_graph_info_mat(sample[0])
                        neighbors_r, attributes_r, u_init_r = get_graph_info_mat(sample[1])
                        sim = sess.run(cos_similarity, {
                            neighbors_left: neighbors_l, attributes_left: attributes_l, u_init_left: u_init_l,
                            neighbors_right: neighbors_r, attributes_right: attributes_r, u_init_right: u_init_r,
                        })
                        if sim > 0 and ground_truth == 1:
                            correct += 1
                        elif sim < 0 and ground_truth == -1:
                            correct += 1

                samples, labels = learning_data['train']['sample'], learning_data['train']['label']
                for sample, ground_truth in zip(samples, labels):
                    # Build neighbors, attributes, and u_init
                    neighbors_l, attributes_l, u_init_l = get_graph_info_mat(sample[0])
                    neighbors_r, attributes_r, u_init_r = get_graph_info_mat(sample[1])

                    _, loss = sess.run([train_op, loss_op], {
                        neighbors_left: neighbors_l, attributes_left: attributes_l, u_init_left: u_init_l,
                        neighbors_right: neighbors_r, attributes_right: attributes_r, u_init_right: u_init_r,
                        label: ground_truth
                    })
                    sys.stdout.write('Epoch: {:10}, Loss: {:15.10f}, Step: {:10}, TestAcc: {:6.10f}    \r'.format(cur_epoch, epoch_loss, cur_step, correct/len(learning_data['test']['sample'])))
                    sys.stdout.flush()
                    cur_step += 1 
                    total_step += 1
                    loss_sum += loss
                epoch_loss = (loss_sum / len(samples))
                cur_epoch += 1

                saver.save(sess, os.path.join(saver_path, 'model.ckpt'), global_step=total_step)


if __name__ == '__main__':
    main(sys.argv)

