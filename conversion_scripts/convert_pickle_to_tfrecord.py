#!/usr/bin/env python3

import networkx as nx
import math
import tensorflow as tf
import os
import sys
import shutil
import csv
import numpy as np
from random import shuffle
import pickle
import argparse
import subprocess
import progressbar
from datetime import datetime
from utils.eval_utils import _start_shell


def get_number_of_attribute():
    from statistical_features import statistical_features
    from structural_features import structural_features
    return len(statistical_features) + len(structural_features)


def normalize_data(samples):
    attr_names = {
            'num_calls': [],
            'num_transfer': [],
            'num_arithmetic': [],
            'num_instructions': [],
            'betweenness_centrality': [],
            'num_offspring': [],
            'num_string': [],
            'num_numeric_constant': []}
     
    for attr_name in attr_names:
        for pair in samples:
            for i in range(2):
                graph = pair[i]['graph']
                for node_id in graph.nodes:
                    attr_names[attr_name].append(graph.nodes[node_id][attr_name])
    attr_avg_std_map = {} 
    for attr_name in attr_names:
        attr_avg_std_map[attr_name] = {}
        attr_avg_std_map[attr_name]['avg'] = np.average(attr_names[attr_name])
        attr_avg_std_map[attr_name]['std'] = np.std(attr_names[attr_name])
        if attr_avg_std_map[attr_name]['std'] == 0:
            attr_avg_std_map[attr_name]['std'] = 1

    return attr_avg_std_map


def n_hot(max_node_num, ids):
    v = np.zeros(max_node_num)
    if ids is not None:
        np.put(v, ids, 1)
    return v


def get_graph_info_mat(graph, attr_avg_std_map, max_node_num, attributes_dim, emb_size):
    graph = graph['graph']
    neighbors = []
    attributes = []

    undir_graph = graph.to_undirected()
    undir_graph = nx.relabel.convert_node_labels_to_integers(undir_graph, first_label=0)

    if max_node_num < len(undir_graph):
        raise ValueError('Number of nodes in graph "{}" is larger than MaxNodeNum: {} >= MaxNodeNum'.format(undir_graph, len(undir_graph)))

    attr_names = ['num_calls', 'num_transfer', 'num_arithmetic', 'num_instructions', 'betweenness_centrality', 'num_offspring', 'num_string', 'num_numeric_constant']
    for idx in range(max_node_num):
        node_id = idx
        if node_id in undir_graph.nodes:
            neighbor_ids = list(undir_graph.neighbors(node_id)) 
            neighbors.append(n_hot(max_node_num, neighbor_ids))
            attrs = []
            for attr_name in attr_names:
                attrs.append((undir_graph.nodes[node_id][attr_name] - attr_avg_std_map[attr_name]['avg']) / attr_avg_std_map[attr_name]['std'])
            attributes.append(attrs)
        else:
            neighbors.append(n_hot(max_node_num, None))
            attributes.append(np.zeros(attributes_dim))
    return neighbors, attributes, np.zeros((max_node_num, emb_size))



def _int64_feature(value):
    """
    generate int64 feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """
    generate float feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
    parser = argparse.ArgumentParser(description='Convert the training pickle data to tfrecord format.')
    parser.add_argument('TrainingDataPlk', help='The pickle format training data.')
    parser.add_argument('OutputDir', help='Path to folder to store output tfrecord files.')
    parser.add_argument('MaxNodeNum', help='Max numbers in nodes of graph.', type=int)
    parser.add_argument('EmbeddingSize', help='Embedding size for the model.', type=int)
    parser.add_argument('TrainingShardNum', help='Number of shard to split datasets into.', type=int)
    parser.add_argument('TestShardNum', help='Number of shard to split datasets into.', type=int)
    args = parser.parse_args()

    attributes_dim = get_number_of_attribute()
    with open(args.TrainingDataPlk, 'rb') as f:
        learning_data = pickle.load(f)
        attr_avg_std_map = normalize_data(learning_data['train']['sample'])

    shard_num = {'train': args.TrainingShardNum, 'test': args.TestShardNum}
    for cur_data_type in ['train', 'test']:
        if len(learning_data[cur_data_type]['sample']) % shard_num[cur_data_type] != 0:
            print('Number of samples in {} % num of shards is not zero. ({} % {} != 0)'.format(cur_data_type, len(learning_data[cur_data_type]['sample']), shard_num[cur_data_type]))
            sys.exit(-1)

    for cur_data_type in ['train', 'test']:
        cur_shard = 0
        cur_sample_idx = 0

        num_sample_per_shard = len(learning_data[cur_data_type]['sample']) / shard_num[cur_data_type]

        for sample, label in zip(learning_data[cur_data_type]['sample'], learning_data[cur_data_type]['label']):
            if cur_sample_idx % num_sample_per_shard == 0:
                writer = tf.python_io.TFRecordWriter(os.path.join(args.OutputDir, "{}-{}-of-{}.tfrecord".format(cur_data_type, cur_shard + 1, shard_num[cur_data_type])))

            neighbors_l, attributes_l, u_init_l = get_graph_info_mat(sample[0], attr_avg_std_map, args.MaxNodeNum, attributes_dim, args.EmbeddingSize)
            neighbors_r, attributes_r, u_init_r = get_graph_info_mat(sample[1], attr_avg_std_map, args.MaxNodeNum, attributes_dim, args.EmbeddingSize)

            features = tf.train.Features(feature={
                "label": _int64_feature([label]),
                "neighbors_l":  _float_feature(np.array(neighbors_l, dtype=np.float64).reshape(-1)),
                "neighbors_r":  _float_feature(np.array(neighbors_r, dtype=np.float64).reshape(-1)),
                "attributes_l": _float_feature(np.array(attributes_l, dtype=np.float64).reshape(-1)),
                "attributes_r": _float_feature(np.array(attributes_r, dtype=np.float64).reshape(-1)),
                "u_init_l":     _float_feature(np.array(u_init_l, dtype=np.float64).reshape(-1)),
                "u_init_r":     _float_feature(np.array(u_init_r, dtype=np.float64).reshape(-1)),
                "neighbors_shape": _int64_feature([args.MaxNodeNum, args.MaxNodeNum]),
                "attributes_shape": _int64_feature([args.MaxNodeNum, attributes_dim]),
                "u_init_shape": _int64_feature([args.MaxNodeNum, args.EmbeddingSize]),
                "identifier_left": _bytes_feature(sample[0]['identifier'].encode('utf-8')),
                "identifier_right": _bytes_feature(sample[1]['identifier'].encode('utf-8')),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

            cur_sample_idx += 1

            if cur_sample_idx % num_sample_per_shard == 0:
                writer.close()
                cur_shard += 1


if __name__ == '__main__':
    main(sys.argv)

