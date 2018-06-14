#!/usr/bin/env python3
import networkx as nx
import argparse
import sys
import pickle
import numpy as np
import re
import os
import sqlite3
import queue
import traceback
import multiprocessing
import subprocess
import time
import progressbar
import traceback
from utils.graph_utils import create_acfg_from_file
from utils.eval_utils import _start_shell


def progressbar_process(q, lock, counter, max_length, done):
    try:
        bar = progressbar.ProgressBar(max_value=0)
        while done.value != 1:
            if max_length.value > bar.max_value:
                bar.max_value = max_length.value
            bar.update(counter.value)
            time.sleep(0.1)
    except Exception as e:
        print(e)
        traceback.print_exc()


def create_acfg_process(q, lock, counter):
    try:
        while True:
            fpath = None
            arch = None
            try:
                fpath = q.get(True, 5)
            except queue.Empty as e:
                return

            try:
                acfg = create_acfg_from_file(fpath, 'x86_64_O3')
            except:
                print('!!! Failed to process {}. !!!'.format(fpath))
                print('Unexpected exception in list_function_names: {}'.format(traceback.format_exc()))
                continue

            path_without_ext = os.path.splitext(fpath)[0]
            acfg_path = path_without_ext + '.acfg.plk'
            with open(acfg_path, 'wb') as f:
                pickle.dump(acfg, f)
            lock.acquire()
            counter.value += 1
            lock.release()
    except Exception as e:
        print(e)
        traceback.print_exc()


def main(argv):
    parser = argparse.ArgumentParser(description='Create ACFG for each dot given by list file parameter and output them as pickle files.')
    parser.add_argument('DotListFile', help='A text file contains a list of dotfile path.')
    parser.add_argument('--NumOfProcesses', type=int, default=10, help='A output sqlite db file to save information about binaries.')
    args = parser.parse_args()

    with open(args.BinaryListFile, 'r') as f:
        lines = f.readlines()
        files = [line.strip('\n') for line in lines if len(line.strip('\n')) != 0]

    manager = multiprocessing.Manager()
    q = manager.Queue()
    counter = manager.Value('i', 0)
    max_length = manager.Value('i', 0)
    done = manager.Value('i', 0)
    lock = manager.Lock()
    processes = []

    num_process = args.NumOfProcesses
    for i in range(num_process):
        p = multiprocessing.Process(target=create_acfg_process, args=(q, lock, counter,))
        p.start()
        processes.append(p)
    progress_p = multiprocessing.Process(target=progressbar_process, args=(q, lock, counter, max_length, done,))
    progress_p.start()

    # Parse each file name pattern to extract arch, binary name(problem id)
    for cfg_path in files:
        # For each dot file of function, transfer it into ACFG
        q.put((cfg_path))
        max_length.value += 1

    for proc in processes:
        proc.join()
    done.value = 1
    progress_p.join()



if __name__ == '__main__':
    main(sys.argv)

