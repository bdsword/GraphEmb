#!/usr/bin/env python3
import itertools
import pickle
import sqlite3
import argparse
import os
import sys
import random
import progressbar
from config import archs


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
    parser.add_argument('--Archs', choices=archs, default=archs, nargs='*', help='Archs to be selected.')
    parser.add_argument('--AcceptMinNodeNum', type=int, help='Minimal number of nodes accepted. (Node number <= this arguments are accepted.)')
    args = parser.parse_args()

    TABLE_NAME = 'flow_graph_acfg'

    conn = sqlite3.connect(args.SQLiteDB)
    cur = conn.cursor()

    cur.execute('SELECT DISTINCT contest FROM {}'.format(TABLE_NAME))
    available_contests = [c[0] for c in cur.fetchall()]

    num_positive = int(input('How many positive pair would you like to generate: '))
    num_negative = int(input('How many negative pair would you like to generate: '))
    num_percent = int(input('How much percent of sample would you like to be the test dataset: '))
    test_percent = num_percent / 100.0

    positive_pool = []

    print('Generate positive samples...')
    bar = progressbar.ProgressBar(max_value=num_positive)

    used_pattern = {}
    random.seed(args.Seed)
    count = 0
    while count < num_positive:
        # Random pick contest
        picked_contest = available_contests[random.randrange(0, len(available_contests))]

        # Random pick one author
        cur.execute('SELECT DISTINCT author FROM {} WHERE contest is "{}"'.format(TABLE_NAME, picked_contest))
        available_authors = [a[0] for a in cur.fetchall()]
        picked_author = available_authors[random.randrange(0, len(available_authors))]

        # Random pick one question
        cur.execute('SELECT DISTINCT question FROM {} WHERE contest is "{}" and author is "{}"'.format(TABLE_NAME, picked_contest, picked_author))
        available_questions = [q[0] for q in cur.fetchall()]
        picked_question = available_questions[random.randrange(0, len(available_questions))]

        # Random pick two architecture
        cur.execute('SELECT DISTINCT arch FROM {} WHERE contest is "{}" and author is "{}" and question is "{}"'.format(TABLE_NAME, picked_contest, picked_author, picked_question))
        available_archs = [a[0] for a in cur.fetchall()]
        available_archs = list(set(available_archs) - (set(archs) - set(args.Archs)))
        picked_arch_ids = random.sample(range(0, len(available_archs) - 1), 2)
        picked_arch_1 = available_archs[picked_arch_ids[0]]
        picked_arch_2 = available_archs[picked_arch_ids[1]]

        # Random pick one function
        cur.execute('SELECT DISTINCT function_name FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}"'.format(TABLE_NAME, picked_contest, picked_author, picked_question, picked_arch_1))
        available_funcs_left = [f[0] for f in cur.fetchall()]
        cur.execute('SELECT DISTINCT function_name FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}"'.format(TABLE_NAME, picked_contest, picked_author, picked_question, picked_arch_2))
        available_funcs_right = [f[0] for f in cur.fetchall()]
        both_contain_fucs = list(set(available_funcs_left) & set(available_funcs_right))
        if len(both_contain_fucs) <= 0:
            continue
        picked_func = both_contain_fucs[random.randrange(0, len(both_contain_fucs))]

        # Select the first record
        cur.execute('SELECT * FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}" and function_name is "{}"'.format(TABLE_NAME, picked_contest, picked_author, picked_question, picked_arch_1, picked_func))
        row = cur.fetchone()
        graph_left = load_graph(row[1])
        data_pattern_left = '{}:{}:{}:{}:{}'.format(picked_contest, picked_author, picked_question, picked_arch_1, picked_func)

        # Select the second record
        cur.execute('SELECT * FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}" and function_name is "{}"'.format(TABLE_NAME, picked_contest, picked_author, picked_question, picked_arch_2, picked_func))
        row = cur.fetchone()
        graph_right = load_graph(row[1])
        data_pattern_right = '{}:{}:{}:{}:{}'.format(picked_contest, picked_author, picked_question, picked_arch_2, picked_func)

        if args.AcceptMinNodeNum and (len(graph_left) < args.AcceptMinNodeNum or len(graph_right) < args.AcceptMinNodeNum):
            continue

        # Check the pattern have not been used
        if '{}_{}'.format(data_pattern_left, data_pattern_right) not in used_pattern and '{}_{}'.format(data_pattern_right, data_pattern_left) not in used_pattern:
            # Append data pair to positive_pool
            positive_pool.append([{'graph': graph_left, 'identifier': data_pattern_left}, {'graph': graph_right, 'identifier': data_pattern_right}]) 
            count += 1
            bar.update(count)
            used_pattern['{}_{}'.format(data_pattern_left, data_pattern_right)] = 1
            used_pattern['{}_{}'.format(data_pattern_right, data_pattern_left)] = 1

    negative_pool = []
    cur.execute('SELECT * FROM {}'.format(TABLE_NAME))
    all_rows = cur.fetchall()
    count = 0
    print('Generate negative samples...')
    bar.max_value = num_negative
    while count < num_negative:
        picked_row_ids = random.sample(range(0, len(all_rows) - 1), 2)
        row_pair = [all_rows[picked_row_ids[0]], all_rows[picked_row_ids[1]]]
        if row_pair[0][3] == row_pair[1][3]:
            continue
        if row_pair[0][2] not in args.Archs or row_pair[1][2] not in args.Archs:
            continue
        graph_left = load_graph(row_pair[0][1])
        graph_right = load_graph(row_pair[1][1])

        if args.AcceptMinNodeNum and (len(graph_left) < args.AcceptMinNodeNum or len(graph_right) < args.AcceptMinNodeNum):
            continue

        data_pattern_left = '{}:{}:{}:{}:{}'.format(row_pair[0][6], row_pair[0][5], row_pair[0][4], row_pair[0][2], row_pair[0][5])
        data_pattern_right = '{}:{}:{}:{}:{}'.format(row_pair[1][6], row_pair[1][5], row_pair[1][4], row_pair[1][2], row_pair[1][5])
        # Check the pattern have not been used
        if '{}_{}'.format(data_pattern_left, data_pattern_right) not in used_pattern and '{}_{}'.format(data_pattern_right, data_pattern_left) not in used_pattern:
            negative_pool.append([{'graph': graph_left, 'identifier': data_pattern_left}, {'graph': graph_right, 'identifier': data_pattern_right}]) 
            count += 1
            bar.update(count)
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
