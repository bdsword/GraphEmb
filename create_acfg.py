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

        # For each dot file of function, transfer it into ACFG
        for fname in os.listdir(functions_folder):
            fpath = os.path.join(functions_folder, fname)
            f_path_parts = os.path.splitext(fpath)
            path_without_ext = f_path_parts[0]
            ext = f_path_parts[1]
            if ext == '.dot':
                function_name = os.path.basename(path_without_ext)
                acfg = create_acfg_from_file(fpath, arch)
                # except Exception as e:
                #     print('!!! Failed to process {}. !!!'.format(fpath))
                #     continue
                acfg_path = path_without_ext + '.acfg.plk'
                with open(acfg_path, 'wb') as f:
                    pickle.dump(acfg, f)
                cur.execute('INSERT INTO {} (binary_path, question, acfg_path, arch, function_name, author, contest) VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}");'
                            .format(TABLE_NAME, binary_path, bin_name, acfg_path, arch, function_name, author_name, contest_name))
                conn.commit()
    conn.close()


if __name__ == '__main__':
    main(sys.argv)

