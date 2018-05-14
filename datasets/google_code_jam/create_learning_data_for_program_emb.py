#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import itertools
import networkx as nx
import pickle
import sqlite3
import argparse
import os
import sys
import random
import progressbar
import time
import re
import traceback
from config import archs
from utils.graph_utils import extract_main_graph
from utils.graph_utils import create_acfg_from_file


def sigma_function(input_l_v, P_n, relu_layer_num, batch_size, max_node_num, embedding_size):
    # N: number of nodes
    output = input_l_v # [B, N, p]
    B, N, p = batch_size, max_node_num, embedding_size
    for idx in range(relu_layer_num):
        if idx > 0:
            output = tf.nn.relu(output)
        output = tf.reshape(output, [B * N, p]) # [B, N, p] -> [B * N, p]
        output = tf.matmul(P_n[idx], output, transpose_b=True) # [p, p] x [B x N, p]^T = [p, B x N]
        output = tf.transpose(output) # [B x N, p]
        output = tf.reshape(output, [B, N, p]) # [B, N, p]
    return output


def build_emb_graph(neighbors, attributes, u_init, attributes_dim, emb_size, T, relu_layer_num):
    # neighbors [B x N x N]
    # attributes [B x N x d]
    # u_init [B x N x p]
    with tf.device('/gpu:0'):
        # Static parameters which are shared by each cfg 
        W1 = tf.get_variable("W1", [attributes_dim, emb_size], initializer=tf.random_normal_initializer(stddev=0.5)) # d x p
        W2 = tf.get_variable("W2", [emb_size, emb_size], initializer=tf.random_normal_initializer(stddev=0.5)) # p x p
        P_n = []
        for idx in range(relu_layer_num):
            P_n.append(tf.get_variable("P_n_{}".format(idx), [emb_size, emb_size], initializer=tf.random_normal_initializer(stddev=0.1)))

        # Dynamic parameters for each cfg
        u_v = u_init

        B, N, p = tf.shape(u_v)[0], tf.shape(u_v)[1], tf.shape(u_v)[2]

        for t in range(T):
            l_vs = tf.matmul(neighbors, u_v) # [B, N, N] x [B, N, p] = [B, N, p]
            sigma_output = sigma_function(l_vs, P_n, relu_layer_num, B, N, p) # [B, N, p]

            # Batch-wised: W1 x attributes
            attributes_reshaped = tf.reshape(attributes, [B * N, attributes_dim])
            W1_mul_attributes = tf.reshape(tf.matmul(W1, attributes_reshaped, transpose_a=True, transpose_b=True), [B, N, p]) # [B, N, p]
            
            sigma_output_add_W1_mul_attributes = tf.add(W1_mul_attributes, sigma_output)            

            u_v = tf.tanh(sigma_output_add_W1_mul_attributes) # [B, N, p]
            # u_v = tf.nn.l2_normalize(u_v, 1)
        u_v_sum = tf.reduce_sum(u_v, 1) # [B, p]
        graph_emb = tf.transpose(tf.matmul(W2, u_v_sum, transpose_b=True)) # [p, p] x [B, p] = [p, B]
        # graph_emb = tf.nn.l2_normalize(graph_emb, 1)
    return graph_emb, W1, W2, P_n, u_v, W1_mul_attributes, sigma_output


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


def get_graph_info_mat(graph, max_node_num, attributes_dim, emb_size):
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
                attrs.append(undir_graph.nodes[node_id][attr_name])
            attributes.append(attrs)
        else:
            neighbors.append(n_hot(max_node_num, None))
            attributes.append(np.zeros(attributes_dim))
    return neighbors, attributes, np.zeros((max_node_num, emb_size))


# End of copy

def load_graph(graph_path):
    graph = None
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


external_symbol_fillcolor = '#ff00ff'
def get_function_list(graph):
    func_names = []
    for node_name in graph.nodes:
        fillcolor = re.findall(r'"(.*)"', graph.nodes[node_name]['fillcolor'])[0]
        function_name = graph.nodes[node_name]['label']
        if fillcolor != external_symbol_fillcolor:
            function_name = re.findall(r'"(.*)\\+l"', function_name)[0]
            func_names.append(function_name)

    return func_names


def build_func_embs(funcs_dot_dict, arch, sess, args, norm_graph_emb_inference, neighbors_test, attributes_test, u_init_test, create_cache=True, cache_path=None):
    func_embs = {}
    for func in funcs_dot_dict:
        func_dot = funcs_dot_dict[func]
        if not os.path.isfile(func_dot):
            print('{} does not exist.'.format(func_dot))
            embs = np.zeros(args.EmbeddingSize)
        else:
            dot_statinfo = os.stat(func_dot)
            if dot_statinfo.st_size == 0:
                embs = np.zeros(args.EmbeddingSize)
            else:
                path_without_ext = os.path.splitext(func_dot)[0]
                acfg_plk = path_without_ext + '.maxnode{}_emb{}.acfg.plk'.format(args.MaxNodeNum, args.EmbeddingSize)

                # Try to create function ACFG pickled file if pickle file do not exist
                if not os.path.isfile(acfg_plk):
                    try:
                        acfg = create_acfg_from_file(func_dot, arch)
                        if create_cache:
                            with open(acfg_plk, 'wb') as f:
                                pickle.dump(acfg, f)
                    except Exception as e:
                        print('!!! Failed to process {}. !!!'.format(func_dot))
                        print('Exception: {}'.format(e))
                        print()
                        continue
                else:
                    with open(acfg_plk, 'rb') as f:
                        acfg = pickle.load(f)

                if len(acfg) <= args.MaxNodeNum:
                    neighbors, attributes, u_init = get_graph_info_mat({'graph': acfg}, args.MaxNodeNum, args.AttrDims, args.EmbeddingSize)
                    embs = sess.run(norm_graph_emb_inference, {neighbors_test: [neighbors], attributes_test: [attributes], u_init_test: [u_init]})[0]
                else:
                    raise IndexError('{} contain funciton {} which has more node num than {}. ({} > {})'.format(row['binary_path'], func, args.MaxNodeNum, len(acfg), args.MaxNodeNum))
        func_embs[func] = embs

    if create_cache and not cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(func_embs, f)
    return func_embs


def embed_func_emb_to_graph(graph, function_embs):
    default_dims = len(function_embs[list(function_embs.keys())[0]])
    for node in graph.nodes:
        func_name = graph.nodes[node]['label'].lstrip('"').rstrip('\\l"')
        if func_name in function_embs:
            graph.nodes[node]['attributes'] = function_embs[func_name]
        else:
            graph.nodes[node]['attributes'] = np.zeros(default_dims)
    return graph


def create_acg_by_row(row, sess, args, norm_graph_emb_inference, neighbors_test, attributes_test, u_init_test, create_cache=True):
    cg_path = os.path.splitext(row['binary_path'])[0] + '.dot'
    funcs_dir = os.path.splitext(row['binary_path'])[0] + '_functions'
    acg_plk = os.path.splitext(row['binary_path'])[0] + '.maxnode{}_emb{}.acg.plk'.format(args.MaxNodeNum, args.EmbeddingSize)
    cache_func_embs_plk = os.path.splitext(row['binary_path'])[0] + '.maxnode{}_emb{}.func_embs.plk'.format(args.MaxNodeNum, args.EmbeddingSize)

    main_graph = extract_main_graph(cg_path)
    if len(main_graph) > args.MaxNodeNum:
        return None

    funcs = get_function_list(main_graph)
    func_dots = {}
    for func in funcs:
        func_dots[func] = os.path.join(funcs_dir, func + '.dot')
    try:
        func_embs = build_func_embs(func_dots, row['arch'], sess, args, norm_graph_emb_inference, neighbors_test, attributes_test, u_init_test, True, cache_func_embs_plk)
        acg = embed_func_emb_to_graph(main_graph, func_embs)
    except IndexError as e:
        print(e)
        return None

    if create_cache:
        with open(acg_plk, 'wb') as f:
            pickle.dump(acg, f)
    return acg


def main(argv):
    parser = argparse.ArgumentParser(description='Slice the whole dataset according to the sqlite fileinto train and test data.')
    parser.add_argument('SQLiteDB', help='Path to the sqlite db file contains information about ACFGs.')
    parser.add_argument('MODEL_DIR', help='The folder to save the model.')
    parser.add_argument('OutputPlk', help='Path to the output pickle file.')
    parser.add_argument('TargetFolder', help='Path to the target folder that contains authors\' dir.')
    parser.add_argument('--PosNum', type=int, help='Number of positive samples.', required=True)
    parser.add_argument('--NegNum', type=int, help='Number of negative samples.', required=True)
    parser.add_argument('--TestPercent', type=int, help='Percentage of testing data.', required=True)
    parser.add_argument('--PositivePool', help='Path to the pickled positive pool file.')
    parser.add_argument('--NegativePool', help='Path to the pickled negative pool file.')
    parser.add_argument('--Seed', type=int, default=0, help='Seed to the random number generator.')
    parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID of the GPU card.')
    parser.add_argument('--TF_LOG_LEVEL', default=3, type=int, help='Environment variable to TF_CPP_MIN_LOG_LEVEL')
    parser.add_argument('--T', type=int, default=5, help='The T parameter in the model.(How many hops to propagate information to.)')
    parser.add_argument('--MaxNodeNum', type=int, default=200, help='The max number of nodes per ACFG.')
    parser.add_argument('--AttrDims', type=int, default=8, help='The number of dimensions of attributes.')
    parser.add_argument('--NumberOfRelu', type=int, default=2, help='The number of relu layer in the sigma function.')
    parser.add_argument('--EmbeddingSize', type=int, default=64, help='The dimension of the embedding vectors.')
    parser.add_argument('--MaxNumModelToKeep', type=int, default=100, help='The number of model to keep in the saver directory.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.TF_LOG_LEVEL)

    if not os.path.isdir(args.MODEL_DIR):
        print('MODEL_DIR folder should be a valid folder.')
        sys.exit(-2)

    with tf.variable_scope("siamese") as scope:
        neighbors_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.MaxNodeNum), name='neighbors_test')
        attributes_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.AttrDims), name='attributes_test')
        u_init_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.EmbeddingSize), name='u_init_test')
        graph_emb_inference, W1_inference, W2_inference, P_n_inference, u_v_inference, W1_mul_X_inference, sigma_output_inference = build_emb_graph(neighbors_test, attributes_test, u_init_test, args.AttrDims, args.EmbeddingSize, args.T, args.NumberOfRelu)
        norm_graph_emb_inference = tf.nn.l2_normalize(graph_emb_inference, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=args.MaxNumModelToKeep)
        states = tf.train.get_checkpoint_state(args.MODEL_DIR)
        saver.restore(sess, states.model_checkpoint_path)


        TABLE_NAME = 'program'

        conn = sqlite3.connect(args.SQLiteDB)
        conn.row_factory = dict_factory
        cur = conn.cursor()

        cur.execute('SELECT DISTINCT contest FROM {}'.format(TABLE_NAME))
        available_contests = [c['contest'] for c in cur.fetchall()]

        '''
        num_positive = int(input('How many positive pair would you like to generate: '))
        num_negative = int(input('How many negative pair would you like to generate: '))
        num_percent = int(input('How much percent of sample would you like to be the test dataset: '))
        '''
        num_positive = args.PosNum
        num_negative = args.NegNum
        num_percent = args.TestPercent
        test_percent = num_percent / 100.0

        random.seed(args.Seed)

        if args.PositivePool:
            with open(args.PositivePool, 'rb') as f:
                positive_pool = pickle.load(f)


        print('Generate positive samples...', end='')
        count = 0
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

        cur.execute('SELECT DISTINCT contest FROM {}'.format(TABLE_NAME))
        available_contests = [c['contest'] for c in cur.fetchall()]

        if not args.PositivePool:
            positive_pool = []
            for contest in available_contests:
                cur.execute('SELECT DISTINCT question FROM {} WHERE contest is "{}" and max_node <= {}'.format(TABLE_NAME, contest, args.MaxNodeNum))
                available_questions = [q['question'] for q in cur.fetchall()]
                for question in available_questions:
                    cur.execute('SELECT DISTINCT author FROM {} WHERE contest is "{}" and question is "{}" and max_node <= {}'.format(TABLE_NAME, contest, question, args.MaxNodeNum))
                    available_authors = [a['author'] for a in cur.fetchall()]
                    for author_pair in itertools.combinations(available_authors, 2):
                        cur.execute('SELECT DISTINCT arch FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and max_node <= {}'.format(TABLE_NAME, contest, author_pair[0], question, args.MaxNodeNum))
                        available_archs_left = [a['arch'] for a in cur.fetchall()]
                        cur.execute('SELECT DISTINCT arch FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and max_node <= {}'.format(TABLE_NAME, contest, author_pair[1], question, args.MaxNodeNum))
                        available_archs_right = [a['arch'] for a in cur.fetchall()]
                        for archs_pair in itertools.product(available_archs_left, available_archs_right):
                            cur.execute('SELECT * FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}" and max_node <= {}'.format(TABLE_NAME, contest, author_pair[0], question, archs_pair[0], args.MaxNodeNum))
                            row_left = cur.fetchone()

                            cur.execute('SELECT * FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}" and max_node <= {}'.format(TABLE_NAME, contest, author_pair[1], question, archs_pair[1], args.MaxNodeNum))
                            row_right = cur.fetchone()

                            acg_left = create_acg_by_row(row_left, sess, args, norm_graph_emb_inference, neighbors_test, attributes_test, u_init_test)
                            if acg_left is None:
                                continue

                            acg_right = create_acg_by_row(row_right, sess, args, norm_graph_emb_inference, neighbors_test, attributes_test, u_init_test)
                            if acg_right is None:
                                continue

                            id_left = '{}:{}:{}:{}'.format(contest, author_pair[0], question, archs_pair[0])
                            id_right = '{}:{}:{}:{}'.format(contest, author_pair[1], question, archs_pair[1])
                            pos_sample = [{'graph': acg_left, 'identifier': id_left}, {'graph': acg_right, 'identifier': id_right}]
                            positive_pool.append(pos_sample)
                            count += 1
                            if count % 100 == 0:
                                with open('positive_program_pool.plk', 'wb') as f:
                                    pickle.dump(positive_pool, f)
                            bar.update(count)
                            break # Only the first arch pair is used. In order to prevent duplicated arch

        if args.NegativePool:
            with open(args.NegativePool, 'rb') as f:
                negative_pool = pickle.load(f)

        if not args.NegativePool:
            negative_pool = []
            for contest in available_contests:
                cur.execute('SELECT DISTINCT question FROM {} WHERE contest is "{}" and max_node <= {}'.format(TABLE_NAME, contest, args.MaxNodeNum))
                available_questions = [q['question'] for q in cur.fetchall()]
                # Generate negative samples
                for question_pair in itertools.combinations(available_questions, 2):
                    cur.execute('SELECT DISTINCT author FROM {} WHERE contest is "{}" and question is "{}" and max_node <= {}'.format(TABLE_NAME, contest, question_pair[0], args.MaxNodeNum))
                    available_authors_q0 = [a['author'] for a in cur.fetchall()]
                    cur.execute('SELECT DISTINCT author FROM {} WHERE contest is "{}" and question is "{}" and max_node <= {}'.format(TABLE_NAME, contest, question_pair[1], args.MaxNodeNum))
                    available_authors_q1 = [a['author'] for a in cur.fetchall()]
                    for author_pair in itertools.product(available_authors_q0, available_authors_q1):
                        cur.execute('SELECT DISTINCT arch FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and max_node <= {}'.format(TABLE_NAME, contest, author_pair[0], question_pair[0], args.MaxNodeNum))
                        available_archs_left = [a['arch'] for a in cur.fetchall()]
                        cur.execute('SELECT DISTINCT arch FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and max_node <= {}'.format(TABLE_NAME, contest, author_pair[1], question_pair[1], args.MaxNodeNum))
                        available_archs_right = [a['arch'] for a in cur.fetchall()]
                        for archs_pair in itertools.product(available_archs_left, available_archs_right):
                            cur.execute('SELECT * FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}" and max_node <= {}'.format(TABLE_NAME, contest, author_pair[0], question_pair[0], archs_pair[0], args.MaxNodeNum))
                            row_left = cur.fetchone()

                            cur.execute('SELECT * FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}" and max_node <= {}'.format(TABLE_NAME, contest, author_pair[1], question_pair[1], archs_pair[1], args.MaxNodeNum))
                            row_right = cur.fetchone()

                            acg_left = create_acg_by_row(row_left, sess, args, norm_graph_emb_inference, neighbors_test, attributes_test, u_init_test)
                            if acg_left is None:
                                continue
                            acg_right = create_acg_by_row(row_right, sess, args, norm_graph_emb_inference, neighbors_test, attributes_test, u_init_test)
                            if acg_right is None:
                                continue

                            id_left = '{}:{}:{}:{}'.format(contest, author_pair[0], question_pair[0], archs_pair[0])
                            id_right = '{}:{}:{}:{}'.format(contest, author_pair[1], question_pair[1], archs_pair[1])
                            neg_sample = [{'graph': acg_left, 'identifier': id_left}, {'graph': acg_right, 'identifier': id_right}]
                            negative_pool.append(neg_sample)
                            count += 1
                            if count % 100 == 0:
                                with open('negative_program_pool.plk', 'wb') as f:
                                    pickle.dump(negative_pool, f)
                            bar.update(count)
                            break


    print('positive_pool: ', len(positive_pool), num_positive)
    print('negative_pool: ', len(negative_pool), num_negative)
    positive_pool = random.sample(positive_pool, num_positive)
    negative_pool = random.sample(negative_pool, num_negative)

    num_train_positive = int(len(positive_pool) * (1.0 - test_percent))
    num_train_negative = int(len(negative_pool) * (1.0 - test_percent))
    train_p_sample = positive_pool[:num_train_positive]
    train_p_label = [1] * num_train_positive
    train_n_sample = negative_pool[:num_train_negative]
    train_n_label = [-1] * num_train_negative

    test_p_sample = positive_pool[num_train_positive:]
    test_p_label = [1] * len(test_p_sample)
    test_n_sample = negative_pool[num_train_negative:]
    test_n_label = [-1] * len(test_n_sample)

    learning_data = {'train': {'sample': train_p_sample + train_n_sample, 'label': train_p_label + train_n_label},
                     'test': {'sample': test_p_sample + test_n_sample, 'label': test_p_label + test_n_label}}
    with open(args.OutputPlk, 'wb') as f:
        pickle.dump(learning_data, f)

    conn.close()

if __name__ == '__main__':
    main(sys.argv)
