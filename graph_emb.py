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
        print('B:', B)
        print('N:', N)
        print('p:', p)
        print('neighbors:', neighbors)

        for t in range(T):
            l_vs = tf.matmul(neighbors, u_v) # [B, N, N] x [B, N, p] = [B, N, p]
            print('l_vs:', l_vs)
            sigma_output = sigma_function(l_vs, P_n, relu_layer_num, B, N, p) # [B, N, p]

            # Batch-wised: W1 x attributes
            attributes_reshaped = tf.reshape(attributes, [B * N, attributes_dim])
            W1_mul_attributes = tf.reshape(tf.matmul(W1, attributes_reshaped, transpose_a=True, transpose_b=True), [B, N, p]) # [B, N, p]
            
            sigma_output_add_W1_mul_attributes = tf.add(W1_mul_attributes, sigma_output)            

            u_v = tf.tanh(sigma_output_add_W1_mul_attributes) # [B, N, p]
            # u_v = tf.nn.l2_normalize(u_v, 1)
        u_v_sum = tf.reduce_sum(u_v, 1) # [B, p]
        graph_emb = tf.transpose(tf.matmul(W2, u_v_sum, transpose_b=True)) # [p, p] x [B, p] = [p, B]
        print('graph_emb:', graph_emb)
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
    np.savetxt(os.path.join(target_dir, 'u_left.csv'), u_left, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'u_right.csv'), u_right, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'W1_X_left.csv'), W1_mul_X_left, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'W1_X_right.csv'), W1_mul_X_right, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'sigma_out_left.csv'), sigma_output_left, delimiter=',')
    np.savetxt(os.path.join(target_dir, 'sigma_out_right.csv'), sigma_output_right, delimiter=',')
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

    for sample in samples:
        neighbors_l, attributes_l, u_init_l = get_graph_info_mat(sample[0], attr_avg_std_map, args.MaxNodeNum, attributes_dim, args.EmbeddingSize)
        neighbors_r, attributes_r, u_init_r = get_graph_info_mat(sample[1], attr_avg_std_map, args.MaxNodeNum, attributes_dim, args.EmbeddingSize)
        neighbors_ls.append(neighbors_l)
        neighbors_rs.append(neighbors_r)
        attributes_ls.append(attributes_l)
        attributes_rs.append(attributes_r)
        u_init_ls.append(u_init_l)
        u_init_rs.append(u_init_r)

    return neighbors_ls, neighbors_rs, attributes_ls, attributes_rs, u_init_ls, u_init_rs


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
            loss_op = tf.reduce_mean(tf.square(cos_similarity - label))

            accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.sign(cos_similarity), label), tf.float32)) / tf.cast(tf.shape(neighbors_left)[0], tf.float32)

            # This is vic's loss function
            # loss_op = (1 + label) * (-0.5 + tf.sigmoid(tf.reduce_mean(tf.squared_difference(graph_emb_left, graph_emb_right)))) + (1 - label) * tf.square(1 + cos_similarity)

            # Operations for debug
            debug_ops = {'W1': W1, 'W2': W2, 'P_n': P_n, 'u_left': u_left, 'u_right': u_right,
                         'W1_mul_X_left': W1_mul_X_left, 'W1_mul_X_right': W1_mul_X_right,
                         'sigma_output_left': sigma_output_left, 'sigma_output_right': sigma_output_right}

        train_op = tf.train.AdamOptimizer(args.LearningRate).minimize(loss_op)
    else: 
        with tf.variable_scope("siamese") as scope:
            # Bulid Inference Graph
            neighbors_test = tf.placeholder(tf.float32, shape=(None, args.MaxNeighborsNum)) # N x MaxNeighborsNum
            attributes_test = tf.placeholder(tf.float32, shape=(None, attributes_dim)) # N x d
            u_init_test = tf.placeholder(tf.float32, shape=(None, args.EmbeddingSize)) # N x p
            graph_emb, W1, W2, P_n, u_v, w1_x, sigma_output = build_emb_graph(neighbors_test, attributes_test, u_init_test, # N x p
                                                                              attributes_dim, args.EmbeddingSize, args.T, args.NumberOfRelu)

    samples, labels = shuffle_data(learning_data['train'])
    neighbors_ls, neighbors_rs, attributes_ls, attributes_rs, u_init_ls, u_init_rs = convert_to_training_data(samples, attr_avg_std_map, args, attributes_dim)

    '''
    neighbors_ls = np.asarray(neighbors_ls)
    neighbors_rs = np.asarray(neighbors_rs)
    attributes_ls = np.asarray(attributes_ls)
    attributes_rs = np.asarray(attributes_rs)
    u_init_ls = np.asarray(u_init_ls)
    u_init_rs = np.asarray(u_init_rs)
    '''

    test_neighbors_ls, test_neighbors_rs, test_attributes_ls, test_attributes_rs, test_u_init_ls, test_u_init_rs = convert_to_training_data(learning_data['test']['sample'], attr_avg_std_map, args, attributes_dim)
    test_labels = learning_data['test']['label']

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
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
            acc = 0

            num_positive = 0

            for cur_epoch in range(num_epoch):
                loss_sum = 0
                cur_step = 0
                correct = 0
                if cur_epoch != 0:
                    acc = sess.run(accuracy, {
                        neighbors_left: test_neighbors_ls, attributes_left: test_attributes_ls, u_init_left: test_u_init_ls,
                        neighbors_right: test_neighbors_rs, attributes_right: test_attributes_rs, u_init_right: test_u_init_rs,
                        label: test_labels
                    })

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
                    
                    sys.stdout.write('Epoch: {:10}, Loss: {:15.10f}, Step: {:10}, TestAcc: {:6.10f}    \r'.format(cur_epoch, epoch_loss, cur_step, acc))
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

                    total_step += len(cur_neighbors_ls)
                    loss_sum += loss
                print()
                epoch_loss = (loss_sum / len(samples))
                cur_epoch += 1
                if args.UpdateModel:
                    saver.save(sess, os.path.join(args.SaverDir, 'model.ckpt'), global_step=total_step)


if __name__ == '__main__':
    main(sys.argv)

