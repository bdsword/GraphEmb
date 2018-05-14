#!/usr/bin/env python3

import os
import sys
import pickle
import networkx as nx
import argparse

import queue
import traceback
import multiprocessing
import subprocess
import time
import progressbar


def pickled_process(q, lock, counter):
    while True:
        try:
            fpath, plk_path = q.get(True, 5)
        except queue.Empty as e:
            return

        try:
            graph = nx.drawing.nx_pydot.read_dot(fpath)
        except:
            continue
        with open(plk_path, 'wb') as f:
            pickle.dump(graph, f)
        counter.value += 1


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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('TargetDir', help='The path to the target directory to process.')
parser.add_argument('--NumOfProcesses', default=10, help='How many processes to run the graph pickling.')
args = parser.parse_args()

manager = multiprocessing.Manager()
q = manager.Queue()
counter = manager.Value('i', 0)
lock = manager.Lock()
p = multiprocessing.Pool()

num_process = args.NumOfProcesses
for i in range(num_process):
    p.apply_async(pickled_process, args=(q, lock, counter,))
p.apply_async(progressbar_process, args=(q, lock, counter,))

for root, dirs, files in os.walk(args.TargetDir):
    for fname in files:
        if os.path.splitext(fname)[1] == '.dot':
            fpath = os.path.join(root, fname)
            plk_path = os.path.join(root, fname + '.nxgraph.plk')
            if os.path.isfile(plk_path):
                continue
            q.put((fpath, plk_path))

p.close()
p.join()
