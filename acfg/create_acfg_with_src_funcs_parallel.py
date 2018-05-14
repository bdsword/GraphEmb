#!/usr/bin/env python3
import networkx as nx
import argparse
import sys
from statistical_features import statistical_features
from structural_features import structural_features
import pickle
import numpy as np
from utils import _start_shell
import re
import os
import sqlite3
import queue
import traceback
import multiprocessing
import subprocess
import time
import progressbar


def progressbar_process(q, lock, counter):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    max_length = -1
    while True:
        if q.qsize() > max_length:
            max_length = q.qsize()
            bar.max_value = max_length
        if q.qsize() == 0:
            break
        bar.update(counter.value)
        time.sleep(0.1)


def get_statistical_features(graph, arch):
    features = dict()
    for feature_func in statistical_features:
        features[feature_func.__name__] = dict()
        for node in graph.nodes:
            code = graph.nodes[node]['label']
            features[feature_func.__name__][node] = float(feature_func(code, arch))
    return features


def get_structural_features(graph):
    features = dict()
    for feature_func in structural_features:
        features[feature_func.__name__] = feature_func(graph)
    return features 


def create_acfg_process(q, lock, sqlite_path, counter):
    TABLE_NAME = 'flow_graph_acfg'
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    while True:
        fpath = None
        arch = None
        try:
            fpath, arch, binary_path, bin_name, function_name, author_name, contest_name = q.get(True, 5)
        except queue.Empty as e:
            cur.close()
            conn.close()
            return

        try:
            acfg = create_acfg_from_file(fpath, arch)
        except:
            print('!!! Failed to process {}. !!!'.format(fpath))
            print('Unexpected exception in list_function_names: {}'.format(traceback.format_exc()))
            continue

        path_without_ext = os.path.splitext(fpath)[0]
        acfg_path = path_without_ext + '.acfg.plk'
        with open(acfg_path, 'wb') as f:
            pickle.dump(acfg, f)
        cur.execute('INSERT INTO {} (binary_path, question, acfg_path, arch, function_name, author, contest) VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}");'
                    .format(TABLE_NAME, binary_path, bin_name, acfg_path, arch, function_name, author_name, contest_name))
        conn.commit()
        counter.value += 1


def create_acfg_from_file(file_path, arch):
    cached_plk = file_path + '.nxgraph.plk'
    if os.path.isfile(cached_plk):
        with open(cached_plk, 'rb') as f:
            graph = pickle.load(f)
    else:
        graph = nx.drawing.nx_pydot.read_dot(file_path)
        with open(cached_plk, 'wb') as f:
            pickle.dump(graph, f)

    features_dict = get_statistical_features(graph, arch)
    for feature_name in features_dict:
        nx.set_node_attributes(graph, features_dict[feature_name], feature_name)

    feature_dict = get_structural_features(graph)
    for feature_name in feature_dict:
        nx.set_node_attributes(graph, feature_dict[feature_name], feature_name)

    return graph


def main(argv):
    parser = argparse.ArgumentParser(description='Create ACFG for each binary given by list file parameter and output them as pickle file.')
    parser.add_argument('BinaryListFile', help='A text file contains a list of binary file path.')
    parser.add_argument('SQLiteFile', help='A output sqlite db file to save information about binaries.')
    parser.add_argument('--NumOfProcesses', type=int, default=10, help='A output sqlite db file to save information about binaries.')
    args = parser.parse_args()

    with open(args.BinaryListFile, 'r') as f:
        lines = f.readlines()
        files = [line.strip('\n') for line in lines if len(line.strip('\n')) != 0]

    TABLE_NAME = 'flow_graph_acfg'
    conn = sqlite3.connect(args.SQLiteFile)
    cur = conn.cursor()
    cur.execute('CREATE TABLE {} (binary_path text, acfg_path text, arch varchar(128), function_name varchar(1024), question varchar(64), author varchar(128), contest varchar(256));'.format(TABLE_NAME))
    cur.execute('CREATE INDEX binary_path ON {}(binary_path);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX arch ON {}(arch);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX function_name ON {}(function_name);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX question ON {}(question);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX author ON {}(author);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX contest ON {}(contest);'.format(TABLE_NAME))
    conn.commit()
    conn.close()

    manager = multiprocessing.Manager()
    q = manager.Queue()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    p = multiprocessing.Pool()

    num_process = args.NumOfProcesses
    for i in range(num_process):
        p.apply_async(create_acfg_process, args=(q, lock, args.SQLiteFile, counter,))
    p.apply_async(progressbar_process, args=(q, lock, counter,))

    # Parse each file name pattern to extract arch, binary name(problem id)
    for binary_path in files:
        author_name = os.path.basename(os.path.abspath(os.path.join(binary_path, os.pardir)))
        contest_name = os.path.basename(os.path.abspath(os.path.join(binary_path, os.pardir, os.pardir)))
        file_name = os.path.basename(binary_path)
        file_name = os.path.splitext(file_name)[0]
        pattern = r'(.+)\.(.+)'
        items = re.findall(pattern, file_name)[0]
        bin_name = items[0]
        arch = items[1]
        functions_folder = os.path.splitext(binary_path)[0] + '_functions'

        func_list_path = os.path.join(functions_folder, 'valid_func_list.txt')
        if not os.path.isfile(func_list_path):
            continue
        with open(func_list_path, 'r') as f:
            valid_funcs = [l.strip().split(' ')[1] for l in f.readlines()]

        # For each dot file of function, transfer it into ACFG
        for function_name in valid_funcs:
            fpath = os.path.join(functions_folder, function_name+ '.dot')
            path_without_ext = os.path.splitext(fpath)[0]
            q.put((fpath, arch, binary_path, bin_name, function_name, author_name, contest_name))
    p.close()
    p.join()


if __name__ == '__main__':
    main(sys.argv)

