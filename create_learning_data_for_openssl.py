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

def load_graph(graph_path):
    graph = None
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def main(argv):
    parser = argparse.ArgumentParser(description='Slice the whole dataset according to the sqlite fileinto train and test data.')
    parser.add_argument('SQLiteDB', help='Path to the sqlite db file contains information about ACFGs.')
    parser.add_argument('OutputPlk', help='Path to the output pickle file.')
    parser.add_argument('--Seed', type=int, default=0, help='Seed to the random number generator.')
    args = parser.parse_args()

    TABLE_NAME = 'flow_graph_acfg'

    conn = sqlite3.connect(args.SQLiteDB)
    cur = conn.cursor()

    cur.execute('SELECT DISTINCT bin_name FROM {}'.format(TABLE_NAME))
    available_bins = [c[0] for c in cur.fetchall()]

    num_positive = int(input('How many positive pair would you like to generate: '))
    num_negative = int(input('How many negative pair would you like to generate: '))
    num_percent = int(input('How much percent of sample would you like to be the test dataset: '))
    test_percent = num_percent / 100.0

    positive_pool = []

    random.seed(args.Seed)

    print('Generate positive samples...')
    bar = progressbar.ProgressBar(max_value=num_positive)

    count = 0
    while count < num_positive:
        # Random pick contest
        picked_bin = available_bins[random.randrange(0, len(available_bins))]

        # Random pick two architecture
        cur.execute('SELECT DISTINCT arch FROM {} WHERE bin_name is "{}"'.format(TABLE_NAME, picked_bin))
        available_archs = [a[0] for a in cur.fetchall()]
        picked_arch_ids = random.sample((0, len(available_archs) - 1), 2)
        picked_arch_1 = available_archs[picked_arch_ids[0]]
        picked_arch_2 = available_archs[picked_arch_ids[1]]

        # Random pick one function
        cur.execute('SELECT DISTINCT function_name FROM {} WHERE bin_name is "{}" and arch is "{}"'.format(TABLE_NAME, picked_bin, picked_arch_1))
        available_funcs_left = [f[0] for f in cur.fetchall()]
        cur.execute('SELECT DISTINCT function_name FROM {} WHERE bin_name is "{}" and arch is "{}"'.format(TABLE_NAME, picked_bin, picked_arch_2))
        available_funcs_right = [f[0] for f in cur.fetchall()]
        both_contain_fucs = list(set(available_funcs_left) & set(available_funcs_right))
        picked_func = both_contain_fucs[random.randrange(0, len(both_contain_fucs))]

        # Select the first record
        cur.execute('SELECT * FROM {} WHERE bin_name is "{}" and arch is "{}" and function_name is "{}"'.format(TABLE_NAME, picked_bin, picked_arch_1, picked_func))
        row = cur.fetchone()
        graph_left = load_graph(row[1])
        data_pattern_left = '{}:{}:{}'.format(picked_bin, picked_arch_1, picked_func)

        # Select the second record
        cur.execute('SELECT * FROM {} WHERE bin_name is "{}" and arch is "{}" and function_name is "{}"'.format(TABLE_NAME, picked_bin, picked_arch_2, picked_func))
        row = cur.fetchone()
        graph_right = load_graph(row[1])
        data_pattern_right = '{}:{}:{}'.format(picked_bin, picked_arch_2, picked_func)

        # Append data pair to positive_pool
        positive_pool.append([{'graph': graph_left, 'identifier': data_pattern_left}, {'graph': graph_right, 'identifier': data_pattern_right}]) 
        count += 1
        bar.update(count)

    negative_pool = []
    cur.execute('SELECT * FROM {}'.format(TABLE_NAME))
    all_rows = cur.fetchall()
    count = 0
    print('Generate negative samples...')
    bar.max_value = num_negative
    while count < num_negative:
        picked_row_ids = random.sample((0, len(all_rows) - 1), 2)
        row_pair = [all_rows[picked_row_ids[0]], all_rows[picked_row_ids[1]]]
        if row_pair[0][3] == row_pair[1][3]:
            continue
        graph_left = load_graph(row_pair[0][1])
        graph_right = load_graph(row_pair[1][1])
        data_pattern_left = '{}:{}:{}'.format(row_pair[0][4], row_pair[0][2], row_pair[0][3])
        data_pattern_right = '{}:{}:{}'.format(row_pair[1][4], row_pair[1][2], row_pair[1][3])
        negative_pool.append([{'graph': graph_left, 'identifier': data_pattern_left}, {'graph': graph_right, 'identifier': data_pattern_right}]) 
        count += 1
        bar.update(count)

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
