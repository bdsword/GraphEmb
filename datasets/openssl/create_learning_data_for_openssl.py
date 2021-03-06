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

    cur.execute('SELECT DISTINCT bin_name FROM {}'.format(TABLE_NAME))
    available_bins = [c['bin_name'] for c in cur.fetchall()]

    num_positive = int(input('How many positive pair would you like to generate: '))
    num_negative = int(input('How many negative pair would you like to generate: '))
    num_percent = int(input('How much percent of sample would you like to be the test dataset: '))
    test_percent = num_percent / 100.0

    positive_pool = []

    random.seed(args.Seed)

    print('Generate positive samples...')
    bar = progressbar.ProgressBar(max_value=num_positive)

    used_pattern = {}

    cur.execute('SELECT DISTINCT bin_name FROM {}'.format(TABLE_NAME))
    available_bins = [b['bin_name'] for b in cur.fetchall()]
    count = 0
    for bin_name in available_bins:
        cur.execute('SELECT DISTINCT function_name FROM {} WHERE bin_name is "{}"'.format(TABLE_NAME, bin_name))
        available_funcs = [f['function_name'] for f in cur.fetchall()]
        for func in available_funcs:
            cur.execute('SELECT DISTINCT arch FROM {} WHERE bin_name is "{}" and function_name is "{}"'.format(TABLE_NAME, bin_name, func))
            available_archs = [a['arch'] for a in cur.fetchall()]
            available_archs = list(set(available_archs) - (set(archs) - set(args.Archs)))
            if len(available_archs) < 2:
                continue

            for arch_pair in itertools.combinations(available_archs, 2):
                cur.execute('SELECT * FROM {} WHERE bin_name is "{}" and function_name is "{}" and arch is "{}"'.format(TABLE_NAME, bin_name, func, arch_pair[0]))
                row = cur.fetchone()
                graph_left = load_graph(row['acfg_path'])
                data_pattern_left = '{}:{}:{}'.format(bin_name, arch_pair[0], func)

                cur.execute('SELECT * FROM {} WHERE bin_name is "{}" and function_name is "{}" and arch is "{}"'.format(TABLE_NAME, bin_name, func, arch_pair[1]))
                row = cur.fetchone()
                graph_right = load_graph(row['acfg_path'])
                data_pattern_right = '{}:{}:{}'.format(bin_name, arch_pair[1], func)

                if args.AcceptMinNodeNum and (len(graph_left) < args.AcceptMinNodeNum or len(graph_right) < args.AcceptMinNodeNum):
                    continue

                if args.AcceptMaxNodeNum and (len(graph_left) > args.AcceptMaxNodeNum or len(graph_right) > args.AcceptMaxNodeNum):
                    continue

                positive_pool.append([{'graph': graph_left, 'identifier': data_pattern_left}, {'graph': graph_right, 'identifier': data_pattern_right}]) 
                count += 1
                if count >= num_positive:
                    break
                bar.update(count)

    if count < num_positive:
        print('Notice: the number of positive samples is smaller than expect')

    negative_pool = []
    cur.execute('SELECT * FROM {}'.format(TABLE_NAME))
    all_rows = cur.fetchall()
    count = 0
    print('Generate negative samples...')
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

        data_pattern_left  = '{}:{}:{}'.format(row_pair[0]['bin_name'], row_pair[0]['arch'], row_pair[0]['function_name'])
        data_pattern_right = '{}:{}:{}'.format(row_pair[1]['bin_name'], row_pair[1]['arch'], row_pair[1]['function_name'])
        # Check the pattern have not been used
        if '{}_{}'.format(data_pattern_left, data_pattern_right) not in used_pattern and '{}_{}'.format(data_pattern_right, data_pattern_left) not in used_pattern:
            negative_pool.append([{'graph': graph_left, 'identifier': data_pattern_left}, {'graph': graph_right, 'identifier': data_pattern_right}]) 
            count += 1
            try:
                bar.update(count)
            except:
                pass
            used_pattern['{}_{}'.format(data_pattern_left, data_pattern_right)] = 1
            used_pattern['{}_{}'.format(data_pattern_right, data_pattern_left)] = 1

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
