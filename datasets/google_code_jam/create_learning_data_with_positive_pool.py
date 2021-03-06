#!/usr/bin/env python3
import itertools
import pickle
import sqlite3
import argparse
import os
import sys
import random
import progressbar
import time
from config import archs


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


def main(argv):
    parser = argparse.ArgumentParser(description='Slice the whole dataset according to the sqlite fileinto train and test data.')
    parser.add_argument('SQLiteDB', help='Path to the sqlite db file contains information about ACFGs.')
    parser.add_argument('PositivePoolPlk', help='Path to the pickle file contains positive samples.')
    parser.add_argument('OutputPlk', help='Path to the output pickle file.')
    parser.add_argument('--Seed', type=int, default=0, help='Seed to the random number generator.')
    parser.add_argument('--Archs', choices=archs, default=archs, nargs='*', help='Archs to be selected.')
    parser.add_argument('--AcceptMinNodeNum', type=int, help='Minimal number of nodes accepted. (Node number >= this arguments are accepted.)')
    parser.add_argument('--AcceptMaxNodeNum', type=int, help='Maximal number of nodes accepted. (Node number <= this arguments are accepted.)')
    args = parser.parse_args()

    TABLE_NAME = 'flow_graph_acfg'

    conn = sqlite3.connect(args.SQLiteDB)
    conn.row_factory = dict_factory
    cur = conn.cursor()

    cur.execute('SELECT DISTINCT contest FROM {}'.format(TABLE_NAME))
    available_contests = [c['contest'] for c in cur.fetchall()]

    num_positive = int(input('How many positive pair would you like to generate: '))
    num_negative = int(input('How many negative pair would you like to generate: '))
    num_percent = int(input('How much percent of sample would you like to be the test dataset: '))
    test_percent = num_percent / 100.0

    with open(args.PositivePoolPlk, 'rb') as f:
        all_positive_samples = pickle.load(f)

    random.seed(args.Seed)

    print('Generate positive samples...', end='')

    positive_pool = random.sample(all_positive_samples, num_positive)

    print('Done')

    used_patterns = {}

    negative_pool = []
    cur.execute('SELECT * FROM {}'.format(TABLE_NAME))
    all_rows = cur.fetchall()
    count = 0
    print('Generate negative samples...')
    bar = progressbar.ProgressBar()
    bar.max_value = num_negative
    while count < num_negative:
        picked_row_ids = random.sample(range(0, len(all_rows)), 2)
        row_pair = [all_rows[picked_row_ids[0]], all_rows[picked_row_ids[1]]]
        if row_pair[0]['function_name'] == row_pair[1]['function_name']:
            continue
        if row_pair[0]['arch'] not in args.Archs or row_pair[1]['arch'] not in args.Archs:
            continue
        graph_left = load_graph(row_pair[0]['acfg_path'])
        graph_right = load_graph(row_pair[1]['acfg_path'])

        if args.AcceptMinNodeNum and (len(graph_left) < args.AcceptMinNodeNum or len(graph_right) < args.AcceptMinNodeNum):
            continue
        if args.AcceptMaxNodeNum and (len(graph_left) > args.AcceptMaxNodeNum or len(graph_right) > args.AcceptMaxNodeNum):
            continue

        data_pattern_left  = '{}:{}:{}:{}:{}'.format(row_pair[0]['contest'], row_pair[0]['author'], row_pair[0]['question'], row_pair[0]['arch'], row_pair[0]['function_name'])
        data_pattern_right = '{}:{}:{}:{}:{}'.format(row_pair[1]['contest'], row_pair[1]['author'], row_pair[1]['question'], row_pair[1]['arch'], row_pair[1]['function_name'])
        # Check the pattern have not been used
        if '{}_{}'.format(data_pattern_left, data_pattern_right) not in used_patterns and '{}_{}'.format(data_pattern_left, data_pattern_right) not in used_patterns:
            negative_pool.append([{'graph': graph_left, 'identifier': data_pattern_left}, {'graph': graph_right, 'identifier': data_pattern_right}])
            count += 1
            bar.update(count)
            used_patterns['{}_{}'.format(data_pattern_left, data_pattern_right)] = 1
            used_patterns['{}_{}'.format(data_pattern_left, data_pattern_right)] = 1

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
