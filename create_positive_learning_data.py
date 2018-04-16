#!/usr/bin/env python3
import itertools
import pickle
import sqlite3
import argparse
import os
import sys
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
    parser.add_argument('TargetFolder', help='Path to the target folder that contains authors\' dir.')
    parser.add_argument('--Archs', choices=archs, default=archs, nargs='*', help='Archs to be selected.')
    parser.add_argument('--AcceptMinNodeNum', type=int, help='Minimal number of nodes accepted. (Node number >= this arguments are accepted.)')
    parser.add_argument('--AcceptMaxNodeNum', type=int, help='Maximal number of nodes accepted. (Node number <= this arguments are accepted.)')
    args = parser.parse_args()

    TABLE_NAME = 'flow_graph_acfg'

    conn = sqlite3.connect(args.SQLiteDB)
    cur = conn.cursor()

    cur.execute('SELECT DISTINCT contest FROM {}'.format(TABLE_NAME))
    available_contests = [c[0] for c in cur.fetchall()]

    positive_pool = []

    print('Generate positive samples...')
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    count = 0
    for contest in available_contests:
        cur.execute('SELECT DISTINCT author FROM {} WHERE contest is "{}"'.format(TABLE_NAME, contest))
        available_authors = [a[0] for a in cur.fetchall()]
        for author in available_authors:
            cur.execute('SELECT DISTINCT question FROM {} WHERE contest is "{}" and author is "{}"'.format(TABLE_NAME, contest, author))
            available_questions = [q[0] for q in cur.fetchall()]
            for question in available_questions:
                cur.execute('SELECT DISTINCT arch FROM {} WHERE contest is "{}" and author is "{}" and question is "{}"'.format(TABLE_NAME, contest, author, question))
                available_archs = [a[0] for a in cur.fetchall()]
                available_archs = list(set(available_archs) - (set(archs) - set(args.Archs)))
                if len(available_archs) < 2:
                    continue
                for arch_pair in itertools.combinations(available_archs, 2):
                    cur.execute('SELECT DISTINCT function_name FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}"'.format(TABLE_NAME, contest, author, question, arch_pair[0]))
                    available_funcs_left = [f[0] for f in cur.fetchall()]
                    cur.execute('SELECT DISTINCT function_name FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}"'.format(TABLE_NAME, contest, author, question, arch_pair[1]))
                    available_funcs_right = [f[0] for f in cur.fetchall()]
                    both_contain_fucs = list(set(available_funcs_left) & set(available_funcs_right))
                    if len(both_contain_fucs) <= 0:
                        continue
                    arch1_list_path = os.path.join(args.TargetFolder, author, question + '.' + arch_pair[0] + '_functions', 'valid_func_list.txt')
                    with open(arch1_list_path, 'r') as f:
                        arch1_valid_funcs = [l.strip().split(' ')[1] for l in f.readlines()]
                    arch2_list_path = os.path.join(args.TargetFolder, author, question + '.' + arch_pair[1] + '_functions', 'valid_func_list.txt')
                    with open(arch2_list_path, 'r') as f:
                        arch2_valid_funcs = [l.strip().split(' ')[1] for l in f.readlines()]
                    for func in both_contain_fucs:
                        if func not in arch1_valid_funcs or func not in arch2_valid_funcs:
                            continue
                        # Select the first record
                        cur.execute('SELECT * FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}" and function_name is "{}"'.format(TABLE_NAME, contest, author, question, arch_pair[0], func))
                        row = cur.fetchone()
                        graph_left = load_graph(row[1])
                        data_pattern_left = '{}:{}:{}:{}:{}'.format(contest, author, question, arch_pair[0], func)
                        # Select the second record
                        cur.execute('SELECT * FROM {} WHERE contest is "{}" and author is "{}" and question is "{}" and arch is "{}" and function_name is "{}"'.format(TABLE_NAME, contest, author, question, arch_pair[1], func))
                        row = cur.fetchone()
                        graph_right = load_graph(row[1])
                        data_pattern_right = '{}:{}:{}:{}:{}'.format(contest, author, question, arch_pair[1], func)

                        if args.AcceptMinNodeNum and (len(graph_left) < args.AcceptMinNodeNum or len(graph_right) < args.AcceptMinNodeNum):
                            continue
                        if args.AcceptMaxNodeNum and (len(graph_left) > args.AcceptMaxNodeNum or len(graph_right) > args.AcceptMaxNodeNum):
                            continue

                        positive_pool.append([{'graph': graph_left, 'identifier': data_pattern_left}, {'graph': graph_right, 'identifier': data_pattern_right}]) 
                        count += 1
                        bar.update(count)

    with open(args.OutputPlk, 'wb') as f:
        pickle.dump(positive_pool, f)

    conn.close()

if __name__ == '__main__':
    main(sys.argv)

