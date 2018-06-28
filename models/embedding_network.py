#!/usr/bin/env python3
import tensorflow as tf


class EmbeddingNetwork:
    def __init__(self, relu_layer_num, max_node_num, embedding_size, attributes_dim, T, trainable=True):
        self.relu_layer_num = relu_layer_num
        self.max_node_num = max_node_num
        self.embedding_size = embedding_size
        self.attributes_dim = attributes_dim
        self.T = T

        self.W1 = tf.get_variable("W1", [self.attributes_dim, self.embedding_size], initializer=tf.random_normal_initializer(stddev=0.1), trainable=trainable) # d x p
        self.W2 = tf.get_variable("W2", [self.embedding_size, self.embedding_size], initializer=tf.random_normal_initializer(stddev=0.1), trainable=trainable) # p x p
        self.P_n = []
        for idx in range(self.relu_layer_num):
            self.P_n.append(tf.get_variable("P_n_{}".format(idx), [self.embedding_size, self.embedding_size],
                                            initializer=tf.random_normal_initializer(stddev=0.1), trainable=trainable))


    def __sigma_function(self, input_l_v, batch_size):
        output = input_l_v # [B, N, p]
        for idx in range(self.relu_layer_num):
            if idx > 0:
                output = tf.nn.relu(output)
            output = tf.reshape(output, [batch_size * self.max_node_num, self.embedding_size]) # [B, N, p] -> [B * N, p]
            output = tf.matmul(self.P_n[idx], output, transpose_b=True) # [p, p] x [B x N, p]^T = [p, B x N]
            output = tf.transpose(output) # [B x N, p]
            output = tf.reshape(output, [batch_size, self.max_node_num, self.embedding_size]) # [B, N, p]
        return output


    def embed(self, neighbors, attributes, u_init):
        # neighbors [B x N x N]
        # attributes [B x N x d]
        # u_init [B x N x p]

        # Dynamic parameters for each cfg
        u_v = u_init

        batch_size = tf.shape(u_v)[0]

        for t in range(self.T):
            l_vs = tf.matmul(neighbors, u_v) # [B, N, N] x [B, N, p] = [B, N, p]
            sigma_output = self.__sigma_function(l_vs, batch_size) # [B, N, p]

            # Batch-wised: W1 x attributes
            attributes_reshaped = tf.reshape(attributes, [batch_size * self.max_node_num, self.attributes_dim])

            u_v = tf.tanh(
                    tf.add(
                        tf.reshape(
                            tf.matmul(self.W1, attributes_reshaped, transpose_a=True, transpose_b=True),
                            [batch_size, self.max_node_num, self.embedding_size]),
                        sigma_output)
                  ) # [B, N, p]
        u_v_sum = tf.reduce_sum(u_v, 1) # [B, p]
        graph_emb = tf.transpose(tf.matmul(self.W2, u_v_sum, transpose_b=True)) # [p, p] x [B, p] = [p, B]
        return graph_emb

