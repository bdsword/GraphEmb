#!/usr/bin/env python3
import networkx as nx
import argparse
import sys
import pickle
import numpy as np
import re
import os
import sqlite3
import progressbar
import time
from utils.graph_utils import read_graph
from utils.graph_utils import create_acfg_from_file


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

    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    counter = 0
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
            try:
                acfg = create_acfg_from_file(fpath, arch)
            except:
                print('!!! Failed to process {}. !!!'.format(fpath))
                continue
            acfg_path = path_without_ext + '.acfg.plk'
            with open(acfg_path, 'wb') as f:
                pickle.dump(acfg, f)
            cur.execute('INSERT INTO {} (binary_path, question, acfg_path, arch, function_name, author, contest) VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}");'
                        .format(TABLE_NAME, binary_path, bin_name, acfg_path, arch, function_name, author_name, contest_name))
            conn.commit()
            counter += 1
            bar.update(counter)
    conn.close()


if __name__ == '__main__':
    main(sys.argv)

