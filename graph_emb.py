#!/usr/bin/env python3

import networkx as nx
import math
import tensorflow as tf
import os
import sys
import shutil
import csv
import numpy as np
from utils import _start_shell
from random import shuffle
import pickle
import argparse
import subprocess
import progressbar


def get_number_of_attribute():
    from statistical_features import statistical_features
    from structural_features import structural_features
    return len(statistical_features) + len(structural_features)


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
        W1 = tf.get_variable("W1", [attributes_dim, emb_size], initializer=tf.random_normal_initializer(stddev=0.1)) # d x p
        W2 = tf.get_variable("W2", [emb_size, emb_size], initializer=tf.random_normal_initializer(stddev=0.1)) # p x p
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


def shuffle_data(dataset):
    data_size = len(dataset['sample'])
    idx_list = [i for i in range(data_size)]
    shuffle(idx_list)
    return np.asarray(dataset['sample'])[idx_list], np.asarray(dataset['label'])[idx_list]


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

    neighbors.append(np.zeros(max_node_num))
    attributes.append(np.zeros(attributes_dim))
    undir_graph = graph.to_undirected()
    undir_graph = nx.relabel.convert_node_labels_to_integers(undir_graph, first_label=1)

    if max_node_num <= len(undir_graph):
        raise ValueError('Number of nodes in graph "{}" is larger than MaxNodeNum: {} >= MaxNodeNum'.format(undir_graph, len(undir_graph)))

    attr_names = ['num_calls', 'num_transfer', 'num_arithmetic', 'num_instructions', 'betweenness_centrality', 'num_offspring', 'num_string', 'num_numeric_constant']
    for idx in range(1, max_node_num):
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


def convert_to_training_data(samples, attr_avg_std_map, args, attributes_dim):
    neighbors_ls = []
    neighbors_rs = []
    attributes_ls = []
    attributes_rs = []
    u_init_ls = []
    u_init_rs = []

    counter = 0
    bar = progressbar.ProgressBar(max_value=len(samples))
    for sample in samples:
        neighbors_l, attributes_l, u_init_l = get_graph_info_mat(sample[0], attr_avg_std_map, args.MaxNodeNum, attributes_dim, args.EmbeddingSize)
        neighbors_r, attributes_r, u_init_r = get_graph_info_mat(sample[1], attr_avg_std_map, args.MaxNodeNum, attributes_dim, args.EmbeddingSize)
        neighbors_ls.append(neighbors_l)
        neighbors_rs.append(neighbors_r)
        attributes_ls.append(attributes_l)
        attributes_rs.append(attributes_r)
        u_init_ls.append(u_init_l)
        u_init_rs.append(u_init_r)
        counter += 1
        bar.update(counter)

    return neighbors_ls, neighbors_rs, attributes_ls, attributes_rs, u_init_ls, u_init_rs


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


def main(argv):
    parser = argparse.ArgumentParser(description='Train the graph embedding network for function flow graph.')
    parser.add_argument('TrainingDataPlk', help='The pickle format training data.')
    parser.add_argument('PackedData', help='Path to store the packed learning data.')
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
    parser.add_argument('--MaxNodeNum', type=int, default=200, help='The max number of nodes per ACFG.')
    parser.add_argument('--NumberOfRelu', type=int, default=2, help='The number of relu layer in the sigma function.')
    parser.add_argument('--EmbeddingSize', type=int, default=64, help='The dimension of the embedding vectors.')
    parser.add_argument('--DebugMatsDir', help='The dimension of the embedding vectors.')
    parser.add_argument('--Debug', dest='Debug', help='Debug mode on.', action='store_true')
    parser.add_argument('--no-Debug', dest='Debug', help='Debug mode off.', action='store_false')
    parser.set_defaults(Debug=False)
    parser.add_argument('--TSNE_Mode', dest='TSNE_Mode', help='T-SNE mode on', action='store_true')
    parser.add_argument('--no-TSNE_Mode', dest='TSNE_Mode', help='T-SNE mode off', action='store_false')
    parser.set_defaults(TSNE_Mode=False)
    parser.add_argument('--ShuffleLearningData', dest='ShuffleLearningData', help='Learning data shuffle mode on', action='store_true')
    parser.add_argument('--no-ShuffleLearningData', dest='ShuffleLearningData', help='Learning data shuffle mode off', action='store_false')
    parser.set_defaults(ShuffleLearningData=False)
    parser.add_argument('--TSNE_InputData', help='Data to generate embedding and do T-SNE.')
    parser.add_argument('--TF_LOG_LEVEL', default=3, type=int, help='Environment variable to TF_CPP_MIN_LOG_LEVEL')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.TF_LOG_LEVEL)

    attributes_dim = get_number_of_attribute()

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

    with open(args.TrainingDataPlk, 'rb') as f:
        learning_data = pickle.load(f)
        attr_avg_std_map = normalize_data(learning_data['train']['sample'])

    print('Building model graph...... ')
    if not args.StartIPython and not args.TSNE_Mode:
        with tf.variable_scope("siamese") as scope:
            # Build Training Graph
            neighbors_left = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.MaxNodeNum)) # B x N x N
            attributes_left = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, attributes_dim)) # B x N x d
            u_init_left = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.EmbeddingSize)) # B x N x p
            graph_emb_left, W1, W2, P_n, u_left, W1_mul_X_left, sigma_output_left = build_emb_graph(neighbors_left, attributes_left, u_init_left, # N x p
                                                                                                    attributes_dim, args.EmbeddingSize, args.T, args.NumberOfRelu)

            scope.reuse_variables()

            neighbors_right = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.MaxNodeNum)) #B x N x N
            attributes_right = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, attributes_dim)) # B x N x d
            u_init_right = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.EmbeddingSize)) # B x N x p
            graph_emb_right, W1, W2, P_n, u_right, W1_mul_X_right, sigma_output_right = build_emb_graph(neighbors_right, attributes_right, u_init_right, # N x p
                                                                                                        attributes_dim, args.EmbeddingSize, args.T, args.NumberOfRelu)

            label = tf.placeholder(tf.float32, shape=(None, ))

            norm_emb_left = tf.nn.l2_normalize(graph_emb_left, 1)
            norm_emb_right = tf.nn.l2_normalize(graph_emb_right, 1)
            cos_similarity = tf.reduce_sum(tf.multiply(norm_emb_left, norm_emb_right), 1)
            # loss_op = tf.reduce_mean(tf.square(cos_similarity - label))

            rad = cos_similarity
            loss_p = (1+label)*tf.cast(tf.less(rad, 0.7),tf.float32)*(1+0.7-rad) # > 45 degree, loss is the degree
            loss_n = (1-label)*tf.cast(tf.greater(rad, 0.7),tf.float32)*(1+rad-0.7) # < 45 degree, loss is 45-degree
            loss_op = tf.reduce_mean( tf.square(loss_p + loss_n) )

            # accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.sign(cos_similarity), label), tf.float32)) / tf.cast(tf.shape(neighbors_left)[0], tf.float32)
            cos_45 = np.cos(np.pi/4) # 1/sqrt(2)
            accuracy = tf.add( tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(cos_similarity, cos_45), tf.float32), label), tf.float32)),\
                        tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.less(cos_similarity, cos_45), tf.float32), tf.negative(label)), tf.float32))) \
                        / tf.cast(tf.shape(neighbors_left)[0], tf.float32)

            # This is vic's loss function
            # loss_op = (1 + label) * (-0.5 + tf.sigmoid(tf.reduce_mean(tf.squared_difference(graph_emb_left, graph_emb_right)))) + (1 - label) * tf.square(1 + cos_similarity)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('loss', loss_op)
            merged = tf.summary.merge_all()

            # Operations for debug
            debug_ops = {'W1': W1, 'W2': W2, 'P_n': P_n, 'u_left': u_left, 'u_right': u_right,
                         'W1_mul_X_left': W1_mul_X_left, 'W1_mul_X_right': W1_mul_X_right,
                         'sigma_output_left': sigma_output_left, 'sigma_output_right': sigma_output_right}

        train_op = tf.train.AdamOptimizer(args.LearningRate).minimize(loss_op)
    else: 
        with tf.variable_scope("siamese") as scope:
            # Bulid Inference Graph
            neighbors_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.MaxNodeNum))
            attributes_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, attributes_dim))
            u_init_test = tf.placeholder(tf.float32, shape=(None, args.MaxNodeNum, args.EmbeddingSize))
            graph_emb, W1, W2, P_n, u_v, W1_mul_X, sigma_output = build_emb_graph(neighbors_test, attributes_test, u_init_test,
                                                                                  attributes_dim, args.EmbeddingSize, args.T, args.NumberOfRelu)
    
    print('Preparing the data for the model...... ')
    if args.TSNE_Mode:
        tsne_data = {'samples': None, 'labels': []}
        tsne_neighbors = []
        tsne_attributes = []
        tsne_u_inits = []
        with open(args.TSNE_InputData, 'rb') as f_in:
            data = pickle.load(f_in)
            for sample in data:
                neighbors, attributes, u_init = get_graph_info_mat(sample, attr_avg_std_map, args.MaxNodeNum, attributes_dim, args.EmbeddingSize)
                tsne_neighbors.append(neighbors)
                tsne_attributes.append(attributes)
                tsne_u_inits.append(u_init)
                tsne_data['labels'].append(sample['identifier'])
    else:
        if args.ShuffleLearningData:
            samples, labels = shuffle_data(learning_data['train'])
        else:
            samples, labels = learning_data['train']['sample'], learning_data['train']['label']
        if os.path.isfile(args.PackedData):
            with open(args.PackedData, 'rb') as f:
                packed_data = pickle.load(f)
                neighbors_ls =       packed_data['neighbors_ls']
                neighbors_rs =       packed_data['neighbors_rs']
                attributes_ls =      packed_data['attributes_ls']
                attributes_rs =      packed_data['attributes_rs']
                u_init_ls =          packed_data['u_init_ls']
                u_init_rs =          packed_data['u_init_rs']
                labels =             packed_data['labels']
                test_neighbors_ls =  packed_data['test_neighbors_ls']
                test_neighbors_rs =  packed_data['test_neighbors_rs']
                test_attributes_ls = packed_data['test_attributes_ls']
                test_attributes_rs = packed_data['test_attributes_rs']
                test_u_init_ls =     packed_data['test_u_init_ls']
                test_u_init_rs =     packed_data['test_u_init_rs']
                test_labels =        packed_data['test_labels']
        else:
            print('\tConverting training data...')
            neighbors_ls, neighbors_rs, attributes_ls, attributes_rs, u_init_ls, u_init_rs = convert_to_training_data(samples, attr_avg_std_map, args, attributes_dim)

            print('\tConverting testing data...')
            test_neighbors_ls, test_neighbors_rs, test_attributes_ls, test_attributes_rs, test_u_init_ls, test_u_init_rs = convert_to_training_data(learning_data['test']['sample'], attr_avg_std_map, args, attributes_dim)
            test_labels = learning_data['test']['label']

            if args.PackedData:
                packed_data = {
                    'neighbors_ls'      : neighbors_ls,
                    'neighbors_rs'      : neighbors_rs,
                    'attributes_ls'     : attributes_ls,
                    'attributes_rs'     : attributes_rs,
                    'u_init_ls'         : u_init_ls,
                    'u_init_rs'         : u_init_rs,
                    'labels'            : labels,
                    'test_neighbors_ls' : test_neighbors_ls,
                    'test_neighbors_rs' : test_neighbors_rs,
                    'test_attributes_ls': test_attributes_ls,
                    'test_attributes_rs': test_attributes_rs,
                    'test_u_init_ls'    : test_u_init_ls,
                    'test_u_init_rs'    : test_u_init_rs,
                    'test_labels'       : test_labels,
                }
                with open(args.PackedData, 'wb') as f:
                    pickle.dump(packed_data, f)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    print('Starting the tensorflow session......')
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train_writer = tf.summary.FileWriter(os.path.join(args.LOG_DIR, 'train'), sess.graph)

        sess.run(init_op)
        
        if args.LoadModel:
            print('Loading the stored model......')
            states = tf.train.get_checkpoint_state(args.MODEL_DIR)
            saver.restore(sess, states.model_checkpoint_path)
        if args.StartIPython:
            _start_shell(locals(), globals())
        elif args.TSNE_Mode:
            print('Start in t-SNE mode (Do embeddings for {})'.format(args.TSNE_InputData))
            count = 0
            embs = []
            while count < len(data):
                if len(data) - count > args.BatchSize:
                    cur_neighbors  = tsne_neighbors [count: count + args.BatchSize]
                    cur_attributes = tsne_attributes[count: count + args.BatchSize]
                    cur_u_inits    = tsne_u_inits   [count: count + args.BatchSize]
                else:
                    cur_neighbors  = tsne_neighbors [count:]
                    cur_attributes = tsne_attributes[count:]
                    cur_u_inits    = tsne_u_inits   [count:]
                embs += sess.run(graph_emb, {neighbors_test: cur_neighbors, attributes_test: cur_attributes, u_init_test: cur_u_inits}).tolist()
                count += len(cur_neighbors)
            tsne_data['samples'] = embs
            emb_plk_path = os.path.join(args.LOG_DIR, 'embeddings.plk')
            print('Writing embeddings.plk file to {}......'.format(emb_plk_path))
            with open(emb_plk_path, 'wb') as f_out:
                pickle.dump(tsne_data, f_out)
            metadata_path = os.path.join(args.LOG_DIR, 'metadata.tsv')
            print('Writing metadata.csv file to {}......'.format(metadata_path))
            with open(metadata_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['dim{}'.format(x) for x in range(args.EmbeddingSize)] + ['label'])
                for idx, emb in enumerate(tsne_data['samples']):
                    csv_writer.writerow(emb + [tsne_data['labels'][idx]])
            print('Generate embedding vectors successfully. To view the visualization, please run:\n$ ./create_tsne_projector.py {} {} YOUR_EMBEDDING_LOG_DIR'.format(emb_plk_path, metadata_path))
        else:
            print('Start in training mode.')
            num_epoch = 1000
            epoch_loss = float('Inf')
            total_step = 0
            train_acc = 0
            test_acc = 0

            num_positive = 0
            print('\tStart training epoch......')
            for cur_epoch in range(num_epoch):
                loss_sum = 0
                cur_step = 0
                correct = 0
                if cur_epoch != 0:
                    train_acc = sess.run(accuracy, {
                        neighbors_left: neighbors_ls[:1000], attributes_left: attributes_ls[:1000], u_init_left: u_init_ls[:1000],
                        neighbors_right: neighbors_rs[:1000], attributes_right: attributes_rs[:1000], u_init_right: u_init_rs[:1000],
                        label: labels[:1000]
                    })
                    test_acc = sess.run(accuracy, {
                        neighbors_left:  test_neighbors_ls, attributes_left : test_attributes_ls, u_init_left : test_u_init_ls,
                        neighbors_right: test_neighbors_rs, attributes_right: test_attributes_rs, u_init_right: test_u_init_rs,
                        label: test_labels
                    })
                if args.Debug:
                    idx = 0
                    for neighbors_l, neighbors_r, attributes_l, attributes_r, u_init_l, u_init_r, ground_truth in zip(neighbors_ls, neighbors_rs, attributes_ls, attributes_rs, u_init_ls, u_init_rs, labels):
                        loss, sim = sess.run([loss_op, cos_similarity], {
                            neighbors_left: [neighbors_l], attributes_left: [attributes_l], u_init_left: [u_init_l],
                            neighbors_right: [neighbors_r], attributes_right: [attributes_r], u_init_right: [u_init_r],
                            label: [ground_truth]
                        })
                        if loss > 3 or (loss < 0.1 and loss > 0):
                            print(loss, samples[idx][0]['identifier'], samples[idx][1]['identifier'])
                            if loss < 0.1 and loss > 0:
                                num_positive += 1
                            if num_positive <= 5 or loss > 3:
                                pattern = str(ground_truth) + '_' + '{:.10E}'.format(loss) + '_' +  samples[idx][0]['identifier'] + '_' + samples[idx][1]['identifier']
                                write_debug_mats(sess, debug_ops, {
                                    neighbors_left: [neighbors_l], attributes_left: [attributes_l], u_init_left: [u_init_l],
                                    neighbors_right: [neighbors_r], attributes_right: [attributes_r], u_init_right: [u_init_r]
                                }, args.DebugMatsDir, samples[idx], pattern)
                        idx += 1


                # samples, labels = learning_data['train']['sample'], learning_data['train']['label']

                while cur_step < len(samples):
                    if len(samples) - cur_step > args.BatchSize:
                        cur_neighbors_ls  = neighbors_ls [cur_step: cur_step + args.BatchSize]
                        cur_neighbors_rs  = neighbors_rs [cur_step: cur_step + args.BatchSize]
                        cur_attributes_ls = attributes_ls[cur_step: cur_step + args.BatchSize]
                        cur_attributes_rs = attributes_rs[cur_step: cur_step + args.BatchSize]
                        cur_u_init_ls     = u_init_ls    [cur_step: cur_step + args.BatchSize]
                        cur_u_init_rs     = u_init_rs    [cur_step: cur_step + args.BatchSize]
                        cur_labels        = labels       [cur_step: cur_step + args.BatchSize]
                    else:
                        cur_neighbors_ls  = neighbors_ls [cur_step:]
                        cur_neighbors_rs  = neighbors_rs [cur_step:]
                        cur_attributes_ls = attributes_ls[cur_step:]
                        cur_attributes_rs = attributes_rs[cur_step:]
                        cur_u_init_ls     = u_init_ls    [cur_step:]
                        cur_u_init_rs     = u_init_rs    [cur_step:]
                        cur_labels        = labels       [cur_step:]

                    _, loss = sess.run([train_op, loss_op], {
                        neighbors_left: cur_neighbors_ls, attributes_left: cur_attributes_ls, u_init_left: cur_u_init_ls,
                        neighbors_right: cur_neighbors_rs, attributes_right: cur_attributes_rs, u_init_right: cur_u_init_rs,
                        label: cur_labels
                    })
                    cur_step += len(cur_neighbors_ls)
                    
                    sys.stdout.write('Epoch: {:10}, BatchLoss: {:15.10f}, Step: {:10}, TrainAcc: {:6.10f}, TestAcc: {:6.10f}    \r'.format(cur_epoch, loss, cur_step, train_acc, test_acc))
                    sys.stdout.flush()

                    summary = sess.run(merged, {
                        neighbors_left: cur_neighbors_ls, attributes_left: cur_attributes_ls, u_init_left: cur_u_init_ls,
                        neighbors_right: cur_neighbors_rs, attributes_right: cur_attributes_rs, u_init_right: cur_u_init_rs,
                        label: cur_labels
                    })
                    train_writer.add_summary(summary, total_step)

                    total_step += len(cur_neighbors_ls)
                    loss_sum += loss
                sys.stdout.write('Epoch: {:10}, EpochLoss: {:15.10f}, Step: {:10}, TrainAcc: {:6.10f}, TestAcc: {:6.10f}    \r'.format(cur_epoch, epoch_loss, cur_step, train_acc, test_acc))
                sys.stdout.flush()
                print()
                epoch_loss = (loss_sum / math.ceil(len(samples) / args.BatchSize))
                cur_epoch += 1
                if args.UpdateModel:
                    saver.save(sess, os.path.join(args.MODEL_DIR, 'model.ckpt'), global_step=total_step)
            print('Training finished.')


if __name__ == '__main__':
    main(sys.argv)

