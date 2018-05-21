#!/usr/bin/env python3
import sqlite3
import argparse
import sys
import re
import os
import glob
import queue
import multiprocessing
import progressbar
import time
from utils.graph_utils import read_graph
import traceback


def progressbar_process(q, lock, counter, max_length):
    try:
        bar = progressbar.ProgressBar(max_value=0)
        while True:
            if max_length.value > bar.max_value:
                bar.max_value = max_length.value
            if q.qsize() == 0:
                break
            bar.update(counter.value)
            time.sleep(0.1)
    except Exception as e:
        print(e)
        traceback.print_exc()


def fetch_program_process(q, lock, sqlite_path, counter):
    try:
        TABLE_NAME = 'program'
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        while True:
            fpath = None
            arch = None
            try:
                binary_path, year, contest_name, author_name, bin_name, arch, functions_folder = q.get(True, 5)
            except queue.Empty as e:
                cur.close()
                conn.close()
                return

            max_node_num = -1
            for fname in os.listdir(functions_folder):
                if os.path.splitext(fname)[1] != '.dot':
                    continue
                try:
                    graph = read_graph(os.path.join(functions_folder, fname))
                except Exception as e:
                    continue
                if max_node_num < len(graph):
                    max_node_num = len(graph)

            cur.execute('INSERT INTO {} (binary_path, year, question, arch, author, contest, max_node) VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}");'
                        .format(TABLE_NAME, binary_path, year, bin_name, arch, author_name, contest_name, max_node_num))
            conn.commit()
            counter.value += 1
    except Exception as e:
        print(e)
        traceback.print_exc()


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
    cur.execute('CREATE TABLE {} (binary_path text, year varchar(16), arch varchar(128), question varchar(64), author varchar(128), contest varchar(256), max_node integer);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX binary_path ON {}(binary_path);'.format(TABLE_NAME))
    cur.execute('CREATE INDEX year ON {}(year);'.format(TABLE_NAME))
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
    max_length = manager.Value('i', 0)
    lock = manager.Lock()
    p = multiprocessing.Pool()

    num_process = args.NumOfProcesses
    for i in range(num_process):
        p.apply_async(fetch_program_process, args=(q, lock, args.SQLiteFile, counter,))
    p.apply_async(progressbar_process, args=(q, lock, counter, max_length,))

    for year in os.listdir(args.TargetFolder):
        year_dir = os.path.join(args.TargetFolder, year)
        for contest_name in os.listdir(year_dir):
            contest_dir = os.path.join(year_dir, contest_name)
            # Parse each file name pattern to extract arch, binary name(problem id)
            for author_name in os.listdir(contest_dir):
                author_dir = os.path.join(contest_dir, author_name)
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
                    if not os.path.isdir(functions_folder):
                        continue

                    q.put((binary_path, year, contest_name, author_name, bin_name, arch, functions_folder))
                    max_length.value += 1
    p.close()
    p.join()


if __name__ == '__main__':
    main(sys.argv)

