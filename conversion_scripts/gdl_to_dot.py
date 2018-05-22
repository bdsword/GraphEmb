#!/usr/bin/env python3
import sys
import os
import argparse
import multiprocessing
import traceback
import time
import queue
import subprocess
import fnmatch
import progressbar


def gdl_to_dot_process(q, lock, counter):
    try:
        while True:
            full_path = None
            dot_file = None
            try:
                full_path, dot_file = q.get(True, 5)
            except queue.Empty as e:
                return

            try:
                subprocess.call(['graph-easy', '--input', full_path, '--output', dot_file])
            except:
                print('!!! Failed to process {}. !!!'.format(full_path))
                print('Unexpected exception in gdl_to_dot_process:\n {}'.format(traceback.format_exc()))
                continue
            lock.acquire()
            counter.value += 1
            lock.release()
    except Exception as e:
        print(e)
        traceback.print_exc()


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


def main(args):
    parser = argparse.ArgumentParser(description='Transfer the .gdl files in the given path into .dot files.')
    parser.add_argument('TargetDir', help='A folder path to be processed.')
    parser.add_argument('--NumOfProcess', type=int, default=5, help='Number of parallel processes.')
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    q = manager.Queue()
    counter = manager.Value('i', 0)
    max_length = manager.Value('i', 0)
    lock = manager.Lock()
    processes = []

    for i in range(args.NumOfProcess):
        p = multiprocessing.Process(target=gdl_to_dot_process, args=(q, lock, counter,))
        p.start()
        processes.append(p)
    p = multiprocessing.Process(target=progressbar_process, args=(q, lock, counter, max_length,))
    p.start()
    processes.append(p)

    for root, dirnames, filenames in os.walk(args.TargetDir):
        for filename in fnmatch.filter(filenames, '*.gdl'):
            full_path = os.path.join(root, filename)
            dot_file = os.path.splitext(full_path)[0] + '.dot'
            q.put((full_path, dot_file))
            max_length.value += 1

    for proc in processes:
        proc.join()


if __name__ == '__main__':
    main(sys.argv)
