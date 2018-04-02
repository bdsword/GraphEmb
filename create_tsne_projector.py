#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import sys
import pickle
import argparse


parser = argparse.ArgumentParser(description='Train the graph embedding network for function flow graph.')
parser.add_argument('EmbeddingsPlk', help='Pickle file to the embeddings.')
parser.add_argument('Metadata', help='Path to the the embeddings metadata.')
parser.add_argument('LOG_DIR', help='LOG_DIR for the tensorboard.')
parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID of the GPU card.')
parser.add_argument('--TF_LOG_LEVEL', default=3, type=int, help='Environment variable to TF_CPP_MIN_LOG_LEVEL')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.TF_LOG_LEVEL)

with open(args.EmbeddingsPlk, 'rb') as f_in:
    data = pickle.load(f_in)
    embeddings = tf.Variable(data['samples'], name='embeddings')

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

print('Start in ViewEmbeddings.')
print('Starting tensorflow session......')
with tf.Session() as sess:
    sess.run(init_op)
    print('\tCreating projector......', end='')
    saver.save(sess, os.path.join(args.LOG_DIR, 'embeddings.ckpt'))
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    embedding.metadata_path = args.Metadata
    summary_writer = tf.summary.FileWriter(args.LOG_DIR)
    projector.visualize_embeddings(summary_writer, config)
    print('Done\n')
    print('You can now launch tensorboard with:\n$ tensorboard --logdir {}'.format(args.LOG_DIR))

