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
                output = subprocess.check_output(['graph-easy', '--input', full_path, '--output', dot_file])
                if output.decode('utf-8', 'ignore') != '':
                    raise subprocess.CalledProcessError('Failed to process: {}'.format(full_path))
            except subprocess.CalledProcessError as e:
                print('!!! Failed to process {}. !!!\n'.format(full_path))
                print('Please check: $ graph-easy --input {} --output {}'.format(full_path, dot_file))
                continue
            except:
                print('!!! Failed to process {}. !!!\n'.format(full_path))
                print('Unexpected exception in gdl_to_dot_process:\n {}'.format(traceback.format_exc()))
                continue
            lock.acquire()
            counter.value += 1
            lock.release()
    except Exception as e:
        print(e)
        traceback.print_exc()


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


def main(args):
    parser = argparse.ArgumentParser(description='Transfer the .gdl files in the given folder(with contest data structure) into .dot files according to a valid function list.')
    parser.add_argument('TargetDir', help='A folder path to be processed.')
    parser.add_argument('--NumOfProcess', type=int, default=5, help='Number of parallel processes.')
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    q = manager.Queue()
    counter = manager.Value('i', 0)
    max_length = manager.Value('i', 0)
    done = manager.Value('i', 0)
    lock = manager.Lock()
    processes = []

    for i in range(args.NumOfProcess):
        p = multiprocessing.Process(target=gdl_to_dot_process, args=(q, lock, counter,))
        p.start()
        processes.append(p)
    progress_p = multiprocessing.Process(target=progressbar_process, args=(q, lock, counter, max_length, done,))
    progress_p.start()

    for author in os.listdir(args.TargetDir):
        author_dir = os.path.join(args.TargetDir, author)
        for fname in os.listdir(author_dir):
            if os.path.splitext(fname)[1] == '.run':
                fpath = os.path.join(author_dir, fname)
                function_dir_path = os.path.join(author_dir, os.path.splitext(fname)[0] + '_functions')
                valid_list_file = os.path.join(function_dir_path, 'valid_func_list.txt')

                if not os.path.isfile(valid_list_file):
                    continue

                with open(valid_list_file, 'r') as f:
                    valid_funcs = [l.strip().split(' ')[1] for l in f.readlines()]
                    for valid_func in valid_funcs:
                        func_gdl_path = os.path.join(function_dir_path, valid_func + '.gdl')
                        func_dot_path = os.path.join(function_dir_path, valid_func + '.dot')
                        if os.path.isfile(func_gdl_path):
                            q.put((func_gdl_path, func_dot_path))
                            max_length.value += 1

    for proc in processes:
        proc.join()
    done.value = 1
    progress_p.join()


if __name__ == '__main__':
    main(sys.argv)

