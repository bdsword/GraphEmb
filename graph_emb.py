#!/usr/bin/env python3

import networkx as nx
import math
import tensorflow as tf
import os
import sys
import shutil
import csv
import numpy as np
from utils.eval_utils import _start_shell
from random import shuffle
import pickle
import argparse
import subprocess
import progressbar
from datetime import datetime
import glob
from models.embedding_network import EmbeddingNetwork
import scipy


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


def write_debug_mats(sess, ops, feed_dict, root_dir, sample_pair, information):
    if not os.path.isdir(root_dir):
        raise ValueError('Argument root_dir should be a valid folder.')

    W1, W2, P_n = sess.run([ops['W1'], ops['W2'], ops['P_n']])
    u_left, W1_mul_X_left, sigma_output_left = sess.run([ops['u_left'], ops['W1_mul_X_left'], ops['sigma_output_left']], feed_dict)
    u_right, W1_mul_X_right, sigma_output_right = sess.run([ops['u_right'], ops['W1_mul_X_right'], ops['sigma_output_right']], feed_dict)

    target_dir = os.path.join(root_dir, information)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    left_dot_path = os.path.join(target_dir, sample_pair[0]['identifier'] + '.dot')
    right_dot_path = os.path.join(target_dir, sample_pair[1]['identifier'] + '.dot')
    nx.drawing.nx_pydot.write_dot(sample_pair[0]['graph'], left_dot_path)
    nx.drawing.nx_pydot.write_dot(sample_pair[1]['graph'], right_dot_path)
    subprocess.check_call(['dot', '-Tpng', '-O', left_dot_path])
    subprocess.check_call(['dot', '-Tpng', '-O', right_dot_path])
    np.savetxt(os.path.join(target_dir, 'W1.csv'), W1, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'W2.csv'), W2, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'u_left.csv'), u_left[0], delimiter=',')
    np.savetxt(os.path.join(target_dir, 'u_right.csv'), u_right[0], delimiter=',')
    np.savetxt(os.path.join(target_dir, 'W1_X_left.csv'), W1_mul_X_left[0], delimiter=',')
    np.savetxt(os.path.join(target_dir, 'W1_X_right.csv'), W1_mul_X_right[0], delimiter=',')
    np.savetxt(os.path.join(target_dir, 'sigma_out_left.csv'), sigma_output_left[0], delimiter=',')
    np.savetxt(os.path.join(target_dir, 'sigma_out_right.csv'), sigma_output_right[0], delimiter=',')
    for i in range(len(P_n)):
        np.savetxt(os.path.join(target_dir, 'P_{}.csv'.format(i)), P_n[i], delimiter=',')
    return


def ask_to_clean_dir(dir_path):
    if len(os.listdir(dir_path)) != 0:
        choice = input('Do you want to delete all the files in the {}? (y/n)'.format(dir_path)).lower()
        if choice == 'y' or choice == 'yes':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            return True
        else:
            print('{} is not empty, it is impossible to update the data inside this folder.'.format(dir_path))
            return False
    return True


def find_tfrecord_for(data_type, search_root):
    return glob.glob('{}/{}*.tfrecord'.format(search_root, data_type))


def parse_example_function(example_proto):
    features = {
                "label":            tf.FixedLenFeature((), dtype=tf.int64),
                "neighbors_shape":  tf.FixedLenFeature((2), dtype=tf.int64),
                "attributes_shape": tf.FixedLenFeature((2), dtype=tf.int64),
                "u_init_shape":     tf.FixedLenFeature((2), dtype=tf.int64),
                "identifier_left":  tf.FixedLenFeature((), dtype=tf.string),
                "identifier_right": tf.FixedLenFeature((), dtype=tf.string),
                "neighbors_l":  tf.VarLenFeature(dtype=tf.float32),
                "neighbors_r":  tf.VarLenFeature(dtype=tf.float32),
                "attributes_l": tf.VarLenFeature(dtype=tf.float32),
                "attributes_r": tf.VarLenFeature(dtype=tf.float32),
                "u_init_l":     tf.VarLenFeature(dtype=tf.float32),
                "u_init_r":     tf.VarLenFeature(dtype=tf.float32),
               }
    parsed_features = tf.parse_single_example(example_proto, features)
    for feature_name in parsed_features:
        if feature_name in ['label', 'neighbors_shape', 'attributes_shape', 'u_init_shape', 'identifier_left', 'identifier_right']:
            continue
        feature_type = feature_name.rstrip('_r').rstrip('_l')
        parsed_features[feature_name] = tf.sparse_tensor_to_dense(parsed_features[feature_name])
        parsed_features[feature_name] = tf.reshape(parsed_features[feature_name], parsed_features[feature_type + '_shape'])
    return parsed_features["neighbors_l"], parsed_features["neighbors_r"], parsed_features["attributes_l"], parsed_features["attributes_r"], parsed_features["u_init_l"], parsed_features["u_init_r"], parsed_features["label"], parsed_features["identifier_left"], parsed_features["identifier_right"]


def main(argv):
    parser = argparse.ArgumentParser(description='Train the graph embedding network for function flow graph.')
    parser.add_argument('TrainingDataDir', help='The path to the directory contains training data.')
    parser.add_argument('MODEL_DIR', help='The folder to save the model.')
    parser.add_argument('LOG_DIR', help='The folder to save the model log.')
    parser.add_argument('--LoadModel', dest='LoadModel', help='Load old model in MODEL_DIR.', action='store_true')
    parser.add_argument('--no-LoadModel', dest='LoadModel', help='Do not load old model in MODEL_DIR.', action='store_false')
    parser.set_defaults(LoadModel=False)
    parser.add_argument('--StartIPython', dest='StartIPython', help='Start IPython shell.', action='store_true')
    parser.add_argument('--no-StartIPython', dest='StartIPython', help='Do not start IPython shell.', action='store_false')
    parser.set_defaults(StartIPython=False)
    parser.add_argument('--UpdateModel', dest='UpdateModel', help='Update the model.', action='store_true')
    parser.add_argument('--no-UpdateModel', dest='UpdateModel', help='Do not update the model.', action='store_false')
    parser.set_defaults(UpdateModel=False)
    parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID of the GPU card.')
    parser.add_argument('--BatchSize', type=int, default=32, help='Number of step per-epoch.')
    parser.add_argument('--LearningRate', type=float, default=0.0001, help='The learning rate for the model.')
    parser.add_argument('--T', type=int, default=5, help='The T parameter in the model.(How many hops to propagate information to.)')
    parser.add_argument('--AttrDims', type=int, default=8, help='The dimensions of the attributes.')
    parser.add_argument('--MaxNodeNum', type=int, default=200, help='The max number of nodes per ACFG.')
    parser.add_argument('--Epochs', type=int, default=1000, help='The number of epochs to run.')
    parser.add_argument('--NumberOfRelu', type=int, default=2, help='The number of relu layer in the sigma function.')
    parser.add_argument('--EmbeddingSize', type=int, default=64, help='The dimension of the embedding vectors.')
    parser.add_argument('--MaxNumModelToKeep', type=int, default=100, help='The number of model to keep in the saver directory.')
    parser.add_argument('--DebugMatsDir', help='The dimension of the embedding vectors.')
    parser.add_argument('--Debug', dest='Debug', help='Debug mode on.', action='store_true')
    parser.add_argument('--no-Debug', dest='Debug', help='Debug mode off.', action='store_false')
    parser.set_defaults(Debug=False)
    parser.add_argument('--Inference', dest='Inference', help='Inference mode on.', action='store_true')
    parser.add_argument('--no-Inference', dest='Inference', help='Inference mode off.', action='store_false')
    parser.set_defaults(Inference=False)
    parser.add_argument('--TSNE_Mode', dest='TSNE_Mode', help='T-SNE mode on', action='store_true')
    parser.add_argument('--no-TSNE_Mode', dest='TSNE_Mode', help='T-SNE mode off', action='store_false')
    parser.set_defaults(TSNE_Mode=False)
    parser.add_argument('--ShuffleLearningData', dest='ShuffleLearningData', help='Learning data shuffle mode on', action='store_true')
    parser.add_argument('--no-ShuffleLearningData', dest='ShuffleLearningData', help='Learning data shuffle mode off', action='store_false')
    parser.set_defaults(ShuffleLearningData=False)
    parser.add_argument('--TSNE_InputData', help='Data to generate embedding and do T-SNE.')
    parser.add_argument('--TrainingPlk', help='Pickled data to calculate attr_avg_std_map.')
    parser.add_argument('--TF_LOG_LEVEL', default=3, type=int, help='Environment variable to TF_CPP_MIN_LOG_LEVEL')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.TF_LOG_LEVEL)

    attributes_dim = args.AttrDims

    if not os.path.isdir(args.MODEL_DIR):
        print('MODEL_DIR folder should be a valid folder.')
        sys.exit(-2)
    elif not args.LoadModel:
        if not ask_to_clean_dir(args.MODEL_DIR):
            sys.exit(-3)
        if not ask_to_clean_dir(args.LOG_DIR):
            sys.exit(-4)

    if args.Debug and not args.DebugMatsDir:
        print('DebugMatsDir should be set when Debug mode is on.')
        sys.exit(-6)

    with tf.device('/cpu:0'):
        shuffle_seed = tf.placeholder(tf.int64, shape=[])
        train_filenames = find_tfrecord_for('train', args.TrainingDataDir)
        dataset = tf.data.TFRecordDataset(train_filenames)
        dataset = dataset.map(parse_example_function, num_parallel_calls=8)
        dataset = dataset.shuffle(buffer_size=10000, seed=shuffle_seed).batch(args.BatchSize)
        dataset = dataset.prefetch(buffer_size=4000)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        test_filenames = find_tfrecord_for('test', args.TrainingDataDir)
        test_dataset = tf.data.TFRecordDataset(test_filenames)
        test_dataset = test_dataset.map(parse_example_function, num_parallel_calls=8)
        test_iterator = test_dataset.make_one_shot_iterator()
        test_next_element = test_iterator.get_next()

    print('Building model graph...... [{}]'.format(str(datetime.now())))
    with tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # Build Training Graph
        neighbors_left = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.MaxNodeNum), name='neighbors_left') # B x N x N
        attributes_left = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, attributes_dim), name='attribute_left') # B x N x d
        u_init_left = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.EmbeddingSize), name='u_init_left') # B x N x p

        neighbors_right = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.MaxNodeNum), name='neighbors_right') #B x N x N
        attributes_right = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, attributes_dim), name='attributes_right') # B x N x d
        u_init_right = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.EmbeddingSize), name='u_init_right') # B x N x p

        neighbors_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.MaxNodeNum), name='neighbors_test')
        attributes_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, attributes_dim), name='attributes_test')
        u_init_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.EmbeddingSize), name='u_init_test')

        label = tf.placeholder(tf.float32, shape=(None, ), name='label')

    with tf.variable_scope("siamese") as scope:
        embedding_network = EmbeddingNetwork(args.NumberOfRelu, args.MaxNodeNum, args.EmbeddingSize, args.AttrDims, args.T)
        graph_emb_left = embedding_network.embed(neighbors_left, attributes_left, u_init_left)
        scope.reuse_variables()
        graph_emb_right = embedding_network.embed(neighbors_right, attributes_right, u_init_right)

        norm_emb_left = tf.nn.l2_normalize(graph_emb_left, 1)
        norm_emb_right = tf.nn.l2_normalize(graph_emb_right, 1)
        cos_similarity = tf.reduce_sum(tf.multiply(norm_emb_left, norm_emb_right), 1)

        # Paper's Loss
        loss_op = tf.reduce_mean(tf.square(cos_similarity - label))
        accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.sign(cos_similarity), label), tf.float32)) / tf.cast(tf.shape(neighbors_left)[0], tf.float32)
        positive_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.gather(tf.sign(cos_similarity), tf.where(tf.equal(label, 1))), 1), tf.float32)) / tf.cast(tf.shape(tf.where(tf.equal(label, 1)))[0], tf.float32)
        positive_num = tf.shape(tf.where(tf.equal(label, 1)))[0]
        negative_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.gather(tf.sign(cos_similarity), tf.where(tf.equal(label, -1))), -1), tf.float32)) / tf.cast(tf.shape(tf.where(tf.equal(label, -1)))[0], tf.float32)
        negative_num = tf.shape(tf.where(tf.equal(label, -1)))[0]
        # End of Paper's Loss

        # Vic's Loss
        '''
        loss_p = (1+label)*tf.cast(tf.equal(tf.mod(global_step,2),1),tf.float32)*(1-cos_similarity) # loss is the degree
        loss_n = (1-label)*tf.cast(tf.equal(tf.mod(global_step,2),0),tf.float32)*tf.cast(tf.greater(cos_similarity, 0.5),tf.float32)*(1+ cos_similarity-0.5)

        # loss_p = (1 + label) * (1 - cos_similarity) # loss is the degree
        # loss_n = (1 - label) * tf.cast(tf.greater(cos_similarity, 0.5), tf.float32) * (1 + cos_similarity - 0.5)
        loss_op = tf.reduce_mean( tf.square(loss_p + loss_n) )
        accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.sign(cos_similarity - 0.5), label), tf.float32)) / tf.cast(tf.shape(neighbors_left)[0], tf.float32)
        positive_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.gather(tf.sign(cos_similarity - 0.5), tf.where(tf.equal(label, 1))), 1), tf.float32)) / tf.cast(tf.shape(tf.where(tf.equal(label, 1)))[0], tf.float32)
        positive_num = tf.shape(tf.where(tf.equal(label, 1)))[0]
        negative_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.gather(tf.sign(cos_similarity - 0.5), tf.where(tf.equal(label, -1))), -1), tf.float32)) / tf.cast(tf.shape(tf.where(tf.equal(label, -1)))[0], tf.float32)
        negative_num = tf.shape(tf.where(tf.equal(label, -1)))[0]
        '''
        # End of Vic's Loss

        # Debug ops
        bingo = tf.cast(tf.equal(tf.sign(cos_similarity), label), tf.float32)
        correct_idx = tf.where(tf.equal(bingo, 1))
        incorrect_idx = tf.where(tf.equal(bingo, 0))

        # Bulid Inference Graph
        graph_emb_inference = embedding_network.embed(neighbors_test, attributes_test, u_init_test)
        norm_graph_emb_inference = tf.nn.l2_normalize(graph_emb_inference, 1)

        # This is vic's loss function
        # loss_op = (1 + label) * (-0.5 + tf.sigmoid(tf.reduce_mean(tf.squared_difference(graph_emb_left, graph_emb_right)))) + (1 - label) * tf.square(1 + cos_similarity)
    with tf.name_scope('Accuracy'):
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('positive_accuracy', positive_accuracy)
        tf.summary.scalar('negative_accuracy', negative_accuracy)
    with tf.name_scope('Cost'):
        tf.summary.scalar('loss', loss_op)
    merged = tf.summary.merge_all()

    train_op = tf.train.AdamOptimizer(args.LearningRate).minimize(loss_op, global_step=global_step)
    # train_op = tf.train.GradientDescentOptimizer(args.LearningRate).minimize(loss_op, global_step=global_step)

    print('Preparing the data for the model...... [{}]'.format(str(datetime.now())))
    if args.TSNE_Mode:
        tsne_data = {'samples': None, 'labels': []}
        tsne_neighbors = []
        tsne_attributes = []
        tsne_u_inits = []
        with open(args.TrainingPlk, 'rb') as f_in:
            data = pickle.load(f_in)
            attr_avg_std_map = normalize_data(data['train']['sample'])
        with open(args.TSNE_InputData, 'rb') as f_in:
            data = pickle.load(f_in)
            for sample in data:
                neighbors, attributes, u_init = get_graph_info_mat(sample, attr_avg_std_map, args.MaxNodeNum, attributes_dim, args.EmbeddingSize)
                tsne_neighbors.append(neighbors)
                tsne_attributes.append(attributes)
                tsne_u_inits.append(u_init)
                tsne_data['labels'].append(sample['identifier'])

    print('Starting the tensorflow session...... [{}]'.format(str(datetime.now())))
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(os.path.join(args.LOG_DIR, 'train'), sess.graph)
        saver = tf.train.Saver(max_to_keep=args.MaxNumModelToKeep)

        if args.LoadModel:
            print('Loading the stored model...... [{}]'.format(str(datetime.now())))
            states = tf.train.get_checkpoint_state(args.MODEL_DIR)
            saver.restore(sess, states.model_checkpoint_path)
        else:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

        if args.Inference:
            sess.run(iterator.initializer, feed_dict={shuffle_seed: 0})
            id_embs = {}
            from numpy import linalg as LA
            while True:
                try:
                    cur_neighbors_ls, cur_neighbors_rs, cur_attributes_ls, cur_attributes_rs, cur_u_init_ls, cur_u_init_rs, cur_labels, identifiers_left, identifiers_right = sess.run(next_element)
                except:
                    break
                embs = sess.run(graph_emb_inference, {neighbors_test: cur_neighbors_ls, attributes_test: cur_attributes_ls, u_init_test: cur_u_init_ls})
                for idx, id_ in enumerate(identifiers_left):
                    id_embs[id_.decode('utf-8')] = embs[idx]
                embs = sess.run(norm_graph_emb_inference, {neighbors_test: cur_neighbors_rs, attributes_test: cur_attributes_rs, u_init_test: cur_u_init_rs})
                for idx, id_ in enumerate(identifiers_right):
                    id_embs[id_.decode('utf-8')] = embs[idx]
            while True:
                choice = input('Please input id for inference: ')
                chosen_emb = id_embs[choice]
                pri = []
                for id_ in id_embs:
                    cos_sim = scipy.spatial.distance.cosine(chosen_emb, id_embs[id_])
                    pri.append([id_, -(cos_sim - 1)])
                pri.sort(key=lambda x: x[1])
                for i in reversed(pri[-10:]):
                    print(i[0], i[1])
                print()

            sys.exit(-1)



        if args.StartIPython:
            _start_shell(locals(), globals())
        elif args.TSNE_Mode:
            print('Start in t-SNE mode (Do embeddings for {}) [{}]'.format(args.TSNE_InputData, str(datetime.now())))
            count = 0
            embs = []
            while count < len(data):
                cur_neighbors  = tsne_neighbors [count: count + args.BatchSize]
                cur_attributes = tsne_attributes[count: count + args.BatchSize]
                cur_u_inits    = tsne_u_inits   [count: count + args.BatchSize]
                embs += sess.run(norm_graph_emb_inference, {neighbors_test: cur_neighbors, attributes_test: cur_attributes, u_init_test: cur_u_inits}).tolist()
                count += len(cur_neighbors)
            tsne_data['samples'] = embs
            emb_plk_path = os.path.join(args.LOG_DIR, 'embeddings.plk')
            print('Writing embeddings.plk file to {}...... [{}]'.format(emb_plk_path, str(datetime.now())))
            with open(emb_plk_path, 'wb') as f_out:
                pickle.dump(tsne_data, f_out)
            metadata_path = os.path.join(args.LOG_DIR, 'metadata.tsv')
            print('Writing metadata.csv file to {}...... [{}]'.format(metadata_path, str(datetime.now())))
            with open(metadata_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['dim{}'.format(x) for x in range(args.EmbeddingSize)] + ['label'])
                for idx, emb in enumerate(tsne_data['samples']):
                    csv_writer.writerow(emb + [tsne_data['labels'][idx]])
            print('Generate embedding vectors successfully. To view the visualization, please run:\n$ ./create_tsne_projector.py {} {} YOUR_EMBEDDING_LOG_DIR'.format(emb_plk_path, metadata_path))
        else:
            print('Start in training mode. [{}]'.format(str(datetime.now())))
            epoch_loss = float('Inf')
            total_step = int(sess.run(global_step))
            train_acc = 0
            test_acc = 0

            test_neighbors_ls  = []
            test_neighbors_rs  = []
            test_attributes_ls = []
            test_attributes_rs = []
            test_u_init_ls     = []
            test_u_init_rs     = []
            test_labels        = []
            print('\tLoading testing dataset.... [{}]'.format(str(datetime.now())))
            while True:
                try:
                    test_neighbors_l, test_neighbors_r, test_attributes_l, test_attributes_r, test_u_init_l, test_u_init_r, test_label, identifiers_left, identifiers_right = sess.run(test_next_element)
                except tf.errors.OutOfRangeError:
                    break
                test_neighbors_ls.append(test_neighbors_l)
                test_neighbors_rs.append(test_neighbors_r)
                test_attributes_ls.append(test_attributes_l)
                test_attributes_rs.append(test_attributes_r)
                test_u_init_ls.append(test_u_init_l)
                test_u_init_rs.append(test_u_init_r)
                test_labels.append(test_label)

            cur_epoch = 0
            epoch_pos_sum = 0
            epoch_pos_num = 0
            epoch_neg_sum = 0
            epoch_neg_num = 0
            if args.Debug:
                false_count = 0
            sess.run(iterator.initializer, feed_dict={shuffle_seed: cur_epoch})
            print('\tStart training epoch...... [{}]'.format(str(datetime.now())))
            while True:
                try:
                    cur_neighbors_ls, cur_neighbors_rs, cur_attributes_ls, cur_attributes_rs, cur_u_init_ls, cur_u_init_rs, cur_labels, identifiers_left, identifiers_right = sess.run(next_element)

                    if args.UpdateModel:
                        _ = sess.run(train_op, {
                            neighbors_left: cur_neighbors_ls, attributes_left: cur_attributes_ls, u_init_left: cur_u_init_ls,
                            neighbors_right: cur_neighbors_rs, attributes_right: cur_attributes_rs, u_init_right: cur_u_init_rs,
                            label: cur_labels
                        })

                    loss, batch_acc, positive_acc, pos_num, negative_acc, neg_num = sess.run([loss_op, accuracy, positive_accuracy, positive_num, negative_accuracy, negative_num], {
                        neighbors_left: cur_neighbors_ls, attributes_left: cur_attributes_ls, u_init_left: cur_u_init_ls,
                        neighbors_right: cur_neighbors_rs, attributes_right: cur_attributes_rs, u_init_right: cur_u_init_rs,
                        label: cur_labels
                    })

                    if pos_num != 0:
                        epoch_pos_sum += (positive_acc * pos_num)
                        epoch_pos_num += pos_num
                    if neg_num != 0:
                        epoch_neg_sum += (negative_acc * neg_num)
                        epoch_neg_num += neg_num

                    if args.Debug:
                        if cur_epoch == 0:
                            for cur_neighbors_l, cur_neighbors_r, cur_attributes_l, cur_attributes_r, cur_u_init_l, cur_u_init_r, cur_label, identifier_left, identifier_right in zip(cur_neighbors_ls, cur_neighbors_rs, cur_attributes_ls, cur_attributes_rs, cur_u_init_ls, cur_u_init_rs, cur_labels, identifiers_left, identifiers_right):
                                cos_sim, loss, batch_acc, positive_acc, negative_acc = sess.run([cos_similarity, loss_op, accuracy, positive_accuracy, negative_accuracy], {
                                    neighbors_left: [cur_neighbors_l], attributes_left: [cur_attributes_l], u_init_left: [cur_u_init_l],
                                    neighbors_right: [cur_neighbors_r], attributes_right: [cur_attributes_r], u_init_right: [cur_u_init_r],
                                    label: [cur_label]
                                })
                                if cur_label != np.sign(cos_sim[0]):
                                    print(cur_label, '#', '{:10.8f}'.format(cos_sim[0]), '#', identifier_left.decode('utf-8'), '#', identifier_right.decode('utf-8'))
                                    false_count += 1
                        else:
                            print('False count: ', false_count)
                            sys.exit(-1)

                    sys.stdout.write('Epoch: {:6}, BatchLoss: {:8.7f}, TotalStep: {:7}, TrainAcc: {:.4f}, PosAcc: {:.4f}, NegAcc: {:.4f}, TestAcc: {:.4f} \r'.format(
                                      cur_epoch  , loss              , total_step     , batch_acc       , positive_acc  , negative_acc  , test_acc))
                    sys.stdout.flush()

                    if args.UpdateModel:
                        summary = sess.run(merged, {
                            neighbors_left: cur_neighbors_ls, attributes_left: cur_attributes_ls, u_init_left: cur_u_init_ls,
                            neighbors_right: cur_neighbors_rs, attributes_right: cur_attributes_rs, u_init_right: cur_u_init_rs,
                            label: cur_labels
                        })
                        train_writer.add_summary(summary, total_step)

                    total_step = int(sess.run(global_step))

                    if args.UpdateModel:
                        if os.path.isabs(args.MODEL_DIR):
                            relative_model_dir = os.path.relpath(args.MODEL_DIR, os.getcwd())
                        saver.save(sess, os.path.join(relative_model_dir, 'model.ckpt'), global_step=global_step)
                except tf.errors.OutOfRangeError:
                    test_acc = sess.run(accuracy, {
                        neighbors_left:  test_neighbors_ls, attributes_left : test_attributes_ls, u_init_left : test_u_init_ls,
                        neighbors_right: test_neighbors_rs, attributes_right: test_attributes_rs, u_init_right: test_u_init_rs,
                        label: test_labels
                    })
                    if args.UpdateModel:
                        test_acc_summary = tf.Summary()
                        test_acc_summary.value.add(tag='Accuracy/test_accuracy', simple_value=test_acc)
                        train_writer.add_summary(test_acc_summary, total_step)
                    print('Epoch: {:6}, BatchLoss: {:8.7f}, TotalStep: {:7}, TrainAcc: {:.4f}, PosAcc: {:.4f}, NegAcc: {:.4f},TestAcc: {:.4f}                   '.format(cur_epoch, loss, total_step, (epoch_pos_sum + epoch_neg_sum) / (epoch_pos_num + epoch_neg_num), epoch_pos_sum / epoch_pos_num, epoch_neg_sum / epoch_neg_num, test_acc))
                    sys.stdout.flush()
                    cur_epoch += 1
                    epoch_pos_sum = 0
                    epoch_neg_sum = 0
                    epoch_pos_num = 0
                    epoch_neg_num = 0
                    if cur_epoch < args.Epochs:
                        sess.run(iterator.initializer, feed_dict={shuffle_seed: cur_epoch})
                    else:
                        print('Training finished. [{}]'.format(str(datetime.now())))
                        break


if __name__ == '__main__':
    main(sys.argv)

