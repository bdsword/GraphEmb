#!/usr/bin/env python3
import sqlite3
import networkx as nx
import argparse
import sys
import re
import os
import glob
import queue
import multiprocessing
import progressbar
import time


def progressbar_process(q, lock, counter, progressbar_close):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    max_length = -1
    while progressbar_close.value != 1:
        if q.qsize() > max_length:
            max_length = q.qsize()
            bar.max_value = max_length
        bar.update(counter.value)
        time.sleep(0.1)


def fetch_program_process(q, lock, sqlite_path, counter):
    try:
        TABLE_NAME = 'program'
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        while True:
            fpath = None
            arch = None
            try:
                binary_path, contest_name, author_name, bin_name, arch, functions_folder = q.get(True, 5)
            except queue.Empty as e:
                cur.close()
                conn.close()
                return

            max_node_num = -1
            for fname in os.listdir(functions_folder):
                if os.path.splitext(fname)[1] != '.dot':
                    continue
                try:
                    graph = nx.drawing.nx_pydot.read_dot(os.path.join(functions_folder, fname))
                except Exception as e:
                    continue
                if max_node_num < len(graph):
                    max_node_num = len(graph)

            cur.execute('INSERT INTO {} (binary_path, question, arch, author, contest, max_node) VALUES ("{}", "{}", "{}", "{}", "{}", "{}");'
                        .format(TABLE_NAME, binary_path, bin_name, arch, author_name, contest_name, max_node_num))
            conn.commit()
            counter.value += 1
    except Exception as e:
        print(e)


def main(argv):
    parser = argparse.ArgumentParser(description='Create SQLite file contains information about .run file of given folder.')
    parser.add_argument('TargetFolder', help='The path of the directory to deal with.')
    parser.add_argument('SQLiteFile', help='A output sqlite db file to save information about binaries.')
    parser.add_argument('--NumOfProcesses', type=int, default=10, help='A output sqlite db file to save information about binaries.')
    args = parser.parse_args()

    if not os.path.isdir(args.TargetFolder):
        print('The TargetFolder is not a valid folder.')
        sys.exit(-1)

    TABLE_NAME = 'program'
    conn = sqlite3.connect(args.SQLiteFile)
    cur = conn.cursor()
    cur.execute('CREATE TABLE {} (binary_path text, arch varchar(128), question varchar(64), author varchar(128), contest varchar(256), max_node integer);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX binary_path ON {}(binary_path);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX arch ON {}(arch);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX question ON {}(question);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX author ON {}(author);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX contest ON {}(contest);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX max_node ON {}(max_node);'.format(TABLE_NAME))
    conn.commit()
    conn.close()

    manager = multiprocessing.Manager()
    q = manager.Queue()
    counter = manager.Value('i', 0)
    progressbar_close = manager.Value('i', 0)
    lock = manager.Lock()
    p = multiprocessing.Pool()

    num_process = args.NumOfProcesses
    for i in range(num_process):
        p.apply_async(fetch_program_process, args=(q, lock, args.SQLiteFile, counter,))
    p.apply_async(progressbar_process, args=(q, lock, counter, progressbar_close,))

    # Parse each file name pattern to extract arch, binary name(problem id)
    for author_name in os.listdir(args.TargetFolder):
        author_dir = os.path.join(args.TargetFolder, author_name)
        for fname in os.listdir(author_dir):
            if os.path.splitext(fname)[1] != '.run':
                continue
            binary_path = os.path.join(author_dir, fname)
            author_name = os.path.basename(os.path.abspath(os.path.join(binary_path, os.pardir)))
            contest_name = os.path.basename(os.path.abspath(os.path.join(binary_path, os.pardir, os.pardir)))
            file_name = os.path.basename(binary_path)
            file_name = os.path.splitext(file_name)[0]
            pattern = r'(.+)\.(.+)'
            items = re.findall(pattern, file_name)[0]
            bin_name = items[0]
            arch = items[1]
            functions_folder = os.path.splitext(binary_path)[0] + '_functions'

            q.put((binary_path, contest_name, author_name, bin_name, arch, functions_folder))
    p.close()
    p.join()
    progressbar_close.value = 1


if __name__ == '__main__':
    main(sys.argv)

