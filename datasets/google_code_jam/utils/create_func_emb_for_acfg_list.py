#!/usr/bin/env python3
import networkx as nx
import argparse
import tensorflow as tf
import sys
import pickle
import numpy as np
import re
import os
import sqlite3
import queue
import traceback
import multiprocessing
import subprocess
import time
import progressbar
import traceback
from utils.graph_utils import create_acfg_from_file
from models.embedding_network import EmbeddingNetwork


def n_hot(max_node_num, ids):
    v = np.zeros(max_node_num)
    if ids is not None:
        np.put(v, ids, 1)
    return v


def get_graph_info_mat(graph, max_node_num, attributes_dim, emb_size):
    graph = graph['graph']
    neighbors = []
    attributes = []

    undir_graph = graph.to_undirected()
    undir_graph = nx.relabel.convert_node_labels_to_integers(undir_graph, first_label=0)

    if max_node_num < len(undir_graph):
        raise ValueError('Number of nodes in graph "{}" is larger than ProgMaxNodeNum: {} >= ProgMaxNodeNum'.format(undir_graph, len(undir_graph)))

    attr_names = ['num_calls', 'num_transfer', 'num_arithmetic', 'num_instructions', 'betweenness_centrality', 'num_offspring', 'num_string', 'num_numeric_constant']
    for idx in range(max_node_num):
        node_id = idx
        if node_id in undir_graph.nodes:
            neighbor_ids = list(undir_graph.neighbors(node_id))
            neighbors.append(n_hot(max_node_num, neighbor_ids))
            attrs = []
            for attr_name in attr_names:
                attrs.append(undir_graph.nodes[node_id][attr_name])
            attributes.append(attrs)
        else:
            neighbors.append(n_hot(max_node_num, None))
            attributes.append(np.zeros(attributes_dim))
    return neighbors, attributes, np.zeros((max_node_num, emb_size))

def main(argv):
    parser = argparse.ArgumentParser(description='Create embeddings plk for each ACFG plk given by list file parameter and output them as pickle files.')
    parser.add_argument('ACFGListFile', help='A text file contains a list of ACFG plk files path.')
    parser.add_argument('MODEL_DIR', help='The folder to save the model.')
    parser.add_argument('--Seed', type=int, default=0, help='Seed to the random number generator.')
    parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID of the GPU card.')
    parser.add_argument('--TF_LOG_LEVEL', default=3, type=int, help='Environment variable to TF_CPP_MIN_LOG_LEVEL')
    parser.add_argument('--T', type=int, default=5, help='The T parameter in the model.(How many hops to propagate information to.)')
    parser.add_argument('--MaxNodeNum', type=int, default=200, help='The max number of nodes per ACFG when doing function CFG embedding.')
    parser.add_argument('--NumberOfRelu', type=int, default=2, help='The number of relu layer in the sigma function.')
    parser.add_argument('--EmbeddingSize', type=int, default=64, help='The dimension of the embedding vectors.')
    parser.add_argument('--AttrDims', type=int, default=8, help='The number of attributes.')
    parser.add_argument('--MaxNumModelToKeep', type=int, default=100, help='The number of model to keep in the saver directory.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.TF_LOG_LEVEL)

    if not os.path.isdir(args.MODEL_DIR):
        print('MODEL_DIR folder should be a valid folder.')
        sys.exit(-2)

    with tf.device('/cpu:0'):
        neighbors_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.MaxNodeNum), name='neighbors_test')
        attributes_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.AttrDims), name='attributes_test')
        u_init_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.EmbeddingSize), name='u_init_test')

    with tf.variable_scope("siamese") as scope:
        embedding_network = EmbeddingNetwork(args.NumberOfRelu, args.MaxNodeNum, args.EmbeddingSize, args.AttrDims, args.T)
        graph_emb_inference = embedding_network.embed(neighbors_test, attributes_test, u_init_test)
        norm_graph_emb_inference = tf.nn.l2_normalize(graph_emb_inference, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=args.MaxNumModelToKeep)
        states = tf.train.get_checkpoint_state(args.MODEL_DIR)
        saver.restore(sess, states.model_checkpoint_path)


        with open(args.ACFGListFile, 'r') as f:
            lines = f.readlines()
            files = [line.strip('\n') for line in lines if len(line.strip('\n')) != 0]

        bar = progressbar.ProgressBar(max_value=len(lines))
        for idx, fpath in enumerate(files):
            with open(fpath, 'rb') as f:
                acfg = pickle.load(f)
                if len(acfg) > args.MaxNodeNum:
                    embs = np.zeros(args.EmbeddingSize)
                else:
                    neighbors, attributes, u_init = get_graph_info_mat({'graph': acfg}, args.MaxNodeNum, args.AttrDims, args.EmbeddingSize)
                    embs = sess.run(norm_graph_emb_inference, {neighbors_test: [neighbors], attributes_test: [attributes], u_init_test: [u_init]})[0]
                plk_path = os.path.splitext(fpath)[0] + '.max{}_emb{}.plk'.format(args.MaxNodeNum, args.EmbeddingSize)
                with open(plk_path, 'wb') as f_out:
                    pickle.dump(embs, f_out)
            bar.update(idx)



if __name__ == '__main__':
    main(sys.argv)

