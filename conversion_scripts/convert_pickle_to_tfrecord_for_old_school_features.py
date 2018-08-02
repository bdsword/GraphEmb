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
        raise ValueError('Number of nodes in graph "{}" is larger than MaxNodeNum: {} >= MaxNodeNum'.format(undir_graph, len(undir_graph)))

    for idx in range(max_node_num):
        node_id = idx
        if node_id in undir_graph.nodes:
            neighbor_ids = list(undir_graph.neighbors(node_id)) 
            neighbors.append(n_hot(max_node_num, neighbor_ids))
            attrs = undir_graph.nodes[node_id]['attributes']
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
    parser.add_argument('TrainingShardNum', help='Number of shard to split datasets into.', type=int)
    parser.add_argument('TestShardNum', help='Number of shard to split datasets into.', type=int)

    args = parser.parse_args()

    with open(args.TrainingDataPlk, 'rb') as f:
        learning_data = pickle.load(f)

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

            features = tf.train.Features(feature={
                "labels": _int64_feature([label]),
                "features": _float_feature(sample['feature']),
                "features_shape": _int64_feature(list(np.shape(sample['feature']))),
                "identifiers": _bytes_feature(sample['identifier'].encode('utf-8')),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

            cur_sample_idx += 1

            if cur_sample_idx % num_sample_per_shard == 0:
                writer.close()
                cur_shard += 1


if __name__ == '__main__':
    main(sys.argv)

