#!/usr/bin/env python3
import networkx as nx
import argparse
import sys
from statistical_features import statistical_features
from structural_features import structural_features
import pickle
import numpy as np
from utils import _start_shell
import progressbar
import time
import re
import os
import sqlite3
import queue
import traceback
import multiprocessing
import subprocess


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
            fpath, arch, binary_path, bin_name, function_name = q.get(True, 5)
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
        cur.execute('INSERT INTO {} (binary_path, bin_name, acfg_path, arch, function_name) VALUES ("{}", "{}", "{}", "{}", "{}");'
                    .format(TABLE_NAME, binary_path, bin_name, acfg_path, arch, function_name))
        conn.commit()
        counter.value += 1


def create_acfg_from_file(file_path, arch):
    graph = nx.drawing.nx_pydot.read_dot(file_path)

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
    cur.execute('CREATE TABLE {} (binary_path text, acfg_path text, arch varchar(128), function_name varchar(1024), bin_name varchar(64));'.format(TABLE_NAME))
    cur.execute('CREATE INDEX binary_path ON {}(binary_path);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX arch ON {}(arch);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX function_name ON {}(function_name);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX question ON {}(bin_name);'.format(TABLE_NAME))
    conn.commit()
    conn.close()

    manager = multiprocessing.Manager()
    q = manager.Queue()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    p = multiprocessing.Pool()

    num_process = args.NumOfProcesses
    for i in range(num_process):
        p.apply_async(create_acfg_process, args=(q, lock, args.SQLiteFile, counter, ))

    arch_maps = {'linux-armv4': 'arm', 'linux-mips32': 'mips',  'linux-x86_64-O0': 'x86_64_O0',  'linux-x86_64-O1': 'x86_64_O1',  'linux-x86_64-O2': 'x86_64_O2',  'linux-x86_64-O3': 'x86_64_O3'}

    p.apply_async(progressbar_process, args=(q, lock, counter))

    # Parse each file name pattern to extract arch, binary name(problem id)
    for binary_path in files:
        arch = os.path.basename(os.path.dirname(binary_path))
        if arch == 'linux-mips32':
            continue
        else:
            arch = arch_maps[arch]
        functions_folder = os.path.splitext(binary_path)[0] + '_functions'
        for fname in os.listdir(functions_folder):
            fpath = os.path.join(functions_folder, fname)
            path_without_ext, ext = os.path.splitext(fpath)
            if ext == '.dot':
                function_name = os.path.basename(fname)
                bin_name = os.path.basename(binary_path)
                q.put((fpath, arch, binary_path, bin_name, function_name))

    p.close()
    p.join()


if __name__ == '__main__':
    main(sys.argv)

