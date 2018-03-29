#!/usr/bin/env python3

import networkx as nx
import tensorflow as tf
import os
import sys
import numpy as np
from utils import _start_shell
from random import shuffle
import pickle
import argparse
import subprocess


def get_number_of_attribute():
    from statistical_features import statistical_features
    from structural_features import structural_features
    return len(statistical_features) + len(structural_features)


def simga_function(input_l_v, P_n, relu_layer_num):
    # N: number of nodes
    output = input_l_v
    for idx in range(relu_layer_num):
        output = tf.matmul(P_n[idx], tf.nn.relu(output), transpose_b=True) # [p, p] x [p, N]= [p, N]
        output = tf.transpose(output) # [N, p]
    return output

def build_emb_graph(neighbors, attributes, u_init, attributes_dim, emb_size, T, relu_layer_num):
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
            sigma_output = simga_function(l_vs, P_n, relu_layer_num) # [N, p]
            w1_x = tf.matmul(W1, attributes, transpose_a=True, transpose_b=True)
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
            # u_v = tf.nn.l2_normalize(u_v, 1)
        graph_emb = tf.transpose(tf.matmul(W2, tf.reshape(tf.reduce_sum(u_v[1:], 0), [-1, emb_size]), transpose_b=True)) # ([p, p] x [p, 1])^T = [p, 1]^T = [1, p]
        # graph_emb = tf.nn.l2_normalize(graph_emb, 1)
    return graph_emb, W1, W2, P_n, u_v, w1_x, sigma_output


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

    return attr_avg_std_map


def get_graph_info_mat(graph, attr_avg_std_map, max_neighbors_num, attributes_dim, emb_size):
    graph = graph['graph']
    neighbors = []
    attributes = []

    neighbors.append(np.zeros(max_neighbors_num))
    attributes.append(np.zeros(attributes_dim))
    undir_graph = graph.to_undirected()
    undir_graph = nx.relabel.convert_node_labels_to_integers(undir_graph, first_label=1)
    for node_id in undir_graph.nodes:
        neighbor_ids = list(undir_graph.neighbors(node_id)) 
        if max_neighbors_num <= len(neighbor_ids):
            raise ValueError('Number of neightbors of node "{}" is larger than MaxNeighborsNum: {} >= MaxNeighborsNum'.format(undir_graph.nodes[node_id], len(neighbor_ids)))
        ns = np.pad(neighbor_ids, (0, max_neighbors_num- len(neighbor_ids)), 'constant', constant_values=0)
        neighbors.append(ns)
        attribute = [(undir_graph.nodes[node_id]['num_calls'] - attr_avg_std_map['num_calls']['avg']) / attr_avg_std_map['num_calls']['std'],
                     (undir_graph.nodes[node_id]['num_transfer'] - attr_avg_std_map['num_transfer']['avg']) / attr_avg_std_map['num_transfer']['std'],
                     (undir_graph.nodes[node_id]['num_arithmetic'] - attr_avg_std_map['num_arithmetic']['avg']) / attr_avg_std_map['num_arithmetic']['std'],
                     (undir_graph.nodes[node_id]['num_instructions'] - attr_avg_std_map['num_instructions']['avg']) / attr_avg_std_map['num_instructions']['std'],
                     (undir_graph.nodes[node_id]['betweenness_centrality'] - attr_avg_std_map['betweenness_centrality']['avg']) / attr_avg_std_map['betweenness_centrality']['std'],
                     (undir_graph.nodes[node_id]['num_offspring'] - attr_avg_std_map['num_offspring']['avg']) / attr_avg_std_map['num_offspring']['std'],
                     (undir_graph.nodes[node_id]['num_string'] - attr_avg_std_map['num_string']['avg']) / attr_avg_std_map['num_string']['std'],
                     (undir_graph.nodes[node_id]['num_numeric_constant'] - attr_avg_std_map['num_numeric_constant']['avg']) / attr_avg_std_map['num_numeric_constant']['std'],
                     ]
        attributes.append(attribute)
    return neighbors, attributes, np.zeros((len(graph.nodes), emb_size))


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
    np.savetxt(os.path.join(target_dir, 'u_left.csv'), u_left, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'u_right.csv'), u_right, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'W1_X_left.csv'), W1_mul_X_left, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'W1_X_right.csv'), W1_mul_X_right, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'sigma_out_left.csv'), sigma_output_left, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'sigma_out_right.csv'), sigma_output_right, delimiter=',')
    for i in range(len(P_n)):
        np.savetxt(os.path.join(target_dir, 'P_{}.csv'.format(i)), P_n[i], delimiter=',')
    return


def main(argv):
    parser = argparse.ArgumentParser(description='Train the graph embedding network for function flow graph.')
    parser.add_argument('TrainingDataPlk', help='The pickle format training data.')
    parser.add_argument('SaverDir', help='The folder to save the model.')
    parser.add_argument('--LoadModel', dest='LoadModel', help='Load old model in SaverDir.', action='store_true')
    parser.add_argument('--no-LoadModel', dest='LoadModel', help='Do not load old model in SaverDir.', action='store_false')
    parser.set_defaults(LoadModel=False)
    parser.add_argument('--StartIPython', dest='StartIPython', help='Start IPython shell.', action='store_true')
    parser.add_argument('--no-StartIPython', dest='StartIPython', help='Do not start IPython shell.', action='store_false')
    parser.set_defaults(StartIPython=False)
    parser.add_argument('--UpdateModel', dest='UpdateModel', help='Update the model.', action='store_true')
    parser.add_argument('--no-UpdateModel', dest='UpdateModel', help='Do not update the model.', action='store_false')
    parser.set_defaults(UpdateModel=False)
    parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID of the GPU card.')
    parser.add_argument('--T', type=int, default=5, help='The T parameter in the model.(How many hops to propagate information to.)')
    parser.add_argument('--MaxNeighborsNum', type=int, default=20, help='The max number of neighbords for a single node.(Limited by my implementation.)')
    parser.add_argument('--NumberOfRelu', type=int, default=2, help='The number of relu layer in the sigma function.')
    parser.add_argument('--EmbeddingSize', type=int, default=64, help='The dimension of the embedding vectors.')
    parser.add_argument('--DebugMatsDir', help='The dimension of the embedding vectors.')
    parser.add_argument('--Debug', dest='Debug', help='Debug mode on.', action='store_true')
    parser.add_argument('--no-Debug', dest='Debug', help='Debug mode off.', action='store_false')
    parser.set_defaults(Debug=False)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU_ID)

    attributes_dim = get_number_of_attribute()

    if not os.path.isdir(args.SaverDir):
        print('Saver folder should be a valid folder.')
        sys.exit(-2)

    if args.Debug and not args.DebugMatsDir:
        print('DebugMatsDir should be set when Debug mode is on.')
        sys.exit(-2)

    with open(args.TrainingDataPlk, 'rb') as f:
        learning_data = pickle.load(f)
        attr_avg_std_map = normalize_data(learning_data['train']['sample'])

    if not args.StartIPython:
        with tf.variable_scope("siamese") as scope:
            # Build Training Graph
            neighbors_left = tf.placeholder(tf.int32, shape=(None, args.MaxNeighborsNum)) # N x MaxNeighborsNum
            attributes_left = tf.placeholder(tf.float32, shape=(None, attributes_dim)) # N x d
            u_init_left = tf.placeholder(tf.float32, shape=(None, args.EmbeddingSize)) # N x p
            graph_emb_left, W1, W2, P_n, u_left, W1_mul_X_left, sigma_output_left = build_emb_graph(neighbors_left, attributes_left, u_init_left, # N x p
                                                                                                    attributes_dim, args.EmbeddingSize, args.T, args.NumberOfRelu)

            scope.reuse_variables()

            neighbors_right = tf.placeholder(tf.int32, shape=(None, args.MaxNeighborsNum)) # N x MaxNeighborsNum
            attributes_right = tf.placeholder(tf.float32, shape=(None, attributes_dim)) # N x d
            u_init_right = tf.placeholder(tf.float32, shape=(None, args.EmbeddingSize)) # N x p
            graph_emb_right, W1, W2, P_n, u_right, W1_mul_X_right, sigma_output_right = build_emb_graph(neighbors_right, attributes_right, u_init_right, # N x p
                                                                                                        attributes_dim, args.EmbeddingSize, args.T, args.NumberOfRelu)

            label = tf.placeholder(tf.float32)

            norm_emb_left = tf.nn.l2_normalize(graph_emb_left, 1)
            norm_emb_right = tf.nn.l2_normalize(graph_emb_right, 1)
            cos_similarity = tf.reduce_sum(tf.multiply(norm_emb_left, norm_emb_right))
            loss_op = tf.square(cos_similarity - label)
            # This is vic's loss function
            # loss_op = (1 + label) * (-0.5 + tf.sigmoid(tf.reduce_mean(tf.squared_difference(graph_emb_left, graph_emb_right)))) + (1 - label) * tf.square(1 + cos_similarity)

            # Operations for debug
            debug_ops = {'W1': W1, 'W2': W2, 'P_n': P_n, 'u_left': u_left, 'u_right': u_right,
                         'W1_mul_X_left': W1_mul_X_left, 'W1_mul_X_right': W1_mul_X_right,
                         'sigma_output_left': sigma_output_left, 'sigma_output_right': sigma_output_right}

        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss_op)
    else: 
        with tf.variable_scope("siamese") as scope:
            # Bulid Inference Graph
            neighbors_test = tf.placeholder(tf.int32, shape=(None, args.MaxNeighborsNum)) # N x MaxNeighborsNum
            attributes_test = tf.placeholder(tf.float32, shape=(None, attributes_dim)) # N x d
            u_init_test = tf.placeholder(tf.float32, shape=(None, args.EmbeddingSize)) # N x p
            graph_emb, W1, W2, P_n, u_v, w1_x, sigma_output = build_emb_graph(neighbors_test, attributes_test, u_init_test, # N x p
                                                                              attributes_dim, args.EmbeddingSize, args.T, args.NumberOfRelu)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(args.SaverDir, sess.graph)
        sess.run(init_op)
        
        if args.LoadModel:
            states = tf.train.get_checkpoint_state(args.SaverDir)
            saver.restore(sess, states.model_checkpoint_path)
        if args.StartIPython:
            _start_shell(locals(), globals())
        else:
            num_epoch = 1000
            epoch_loss = float('Inf')
            total_step = 0

            num_positive = 0

            for cur_epoch in range(num_epoch):
                loss_sum = 0
                cur_step = 0
                correct = 0
                samples, labels = learning_data['test']['sample'], learning_data['test']['label']
                if cur_epoch != 0:
                    for sample, ground_truth in zip(samples, labels):
                        neighbors_l, attributes_l, u_init_l = get_graph_info_mat(sample[0], attr_avg_std_map, args.MaxNeighborsNum, attributes_dim, args.EmbeddingSize)
                        neighbors_r, attributes_r, u_init_r = get_graph_info_mat(sample[1], attr_avg_std_map, args.MaxNeighborsNum, attributes_dim, args.EmbeddingSize)
                        sim = sess.run(cos_similarity, {
                            neighbors_left: neighbors_l, attributes_left: attributes_l, u_init_left: u_init_l,
                            neighbors_right: neighbors_r, attributes_right: attributes_r, u_init_right: u_init_r,
                        })
                        if sim > 0 and ground_truth == 1:
                            correct += 1
                        elif sim < 0 and ground_truth == -1:
                            correct += 1

                samples, labels = shuffle_data(learning_data['train'])
                # samples, labels = learning_data['train']['sample'], learning_data['train']['label']

                for sample, ground_truth in zip(samples, labels):
                    # Build neighbors, attributes, and u_init
                    neighbors_l, attributes_l, u_init_l = get_graph_info_mat(sample[0], attr_avg_std_map, args.MaxNeighborsNum, attributes_dim, args.EmbeddingSize)
                    neighbors_r, attributes_r, u_init_r = get_graph_info_mat(sample[1], attr_avg_std_map, args.MaxNeighborsNum, attributes_dim, args.EmbeddingSize)

                    _, loss = sess.run([train_op, loss_op], {
                        neighbors_left: neighbors_l, attributes_left: attributes_l, u_init_left: u_init_l,
                        neighbors_right: neighbors_r, attributes_right: attributes_r, u_init_right: u_init_r,
                        label: ground_truth
                    })
                    sys.stdout.write('Epoch: {:10}, Loss: {:15.10f}, Step: {:10}, TestAcc: {:6.10f}    \r'.format(cur_epoch, epoch_loss, cur_step, correct/len(learning_data['test']['sample'])))
                    sys.stdout.flush()

                    if args.Debug and (loss > 7 or (loss < 0.1 and loss > 0)):
                        if loss < 0.1 and loss > 0:
                            num_positive += 1
                        if num_positive <= 5 or loss > 7:
                            pattern = str(ground_truth) + '_' + '{:.10E}'.format(loss) + '_' +  sample[0]['identifier'] + '_' + sample[1]['identifier']
                            write_debug_mats(sess, debug_ops, {
                                neighbors_left: neighbors_l, attributes_left: attributes_l, u_init_left: u_init_l,
                                neighbors_right: neighbors_r, attributes_right: attributes_r, u_init_right: u_init_r
                            }, args.DebugMatsDir, sample, pattern)

                    loss_summary = tf.Summary()
                    loss_summary.value.add(tag='Loss', simple_value=epoch_loss)
                    acc_summary = tf.Summary()
                    acc_summary.value.add(tag='Accuracy', simple_value=correct/len(learning_data['test']['sample']))
                    train_writer.add_summary(acc_summary, global_step=total_step)
                    train_writer.add_summary(loss_summary, global_step=total_step)

                    cur_step += 1 
                    total_step += 1
                    loss_sum += loss
                print()
                epoch_loss = (loss_sum / len(samples))
                cur_epoch += 1
                if args.UpdateModel:
                    saver.save(sess, os.path.join(args.SaverDir, 'model.ckpt'), global_step=total_step)


if __name__ == '__main__':
    main(sys.argv)

