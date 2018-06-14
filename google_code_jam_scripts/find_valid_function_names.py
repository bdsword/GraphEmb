#!/usr/bin/env python3
import time
import sys
import os
import re
import multiprocessing
import subprocess
import argparse
import queue
import traceback


def list_function_names(q, lock):
    while True:
        try:
            path = None
            try:
                path = q.get(True, 5)
            except queue.Empty as e:
                return
            if path and len(path) > 0:
                print('Processing... {}'.format(path))
                base, ext = os.path.splitext(path)
                if ext == '.cpp':
                    kind = '--c++-kinds=f'
                else:
                    kind = '--c-kinds=f'
                output = subprocess.check_output(['ctags', '-x', kind, path])
                output = output.decode('utf-8', 'ignore')
                func_prototype_map = {}
                for line in output.split('\n'):
                    line = line.strip()
                    if len(line) <= 0:
                        continue
                    # main             function     83 /home/bdsword/Documents/Testbed/FivePeopleTest/2011aad/2.cpp int main()
                    func_name, _, _, _, func_prototype= re.findall(r'^(.+)\s+function\s+\d+\s+((\~|\/[^\/]*)+\/?)(\.cpp|\.cc|\.c)\s+(.*)$', line)[0]
                    func_name = func_name.strip()
                    # mangled_out = subprocess.check_output(['bash', '-c', 'echo "{}" | g++ -x c++ -S - -o- | grep "^_.*:$" | sed -e "s/:$//"'.format(func_prototype)])
                    # mangled_out = mangled_out.decode('utf-8', 'ignore')
                    func_prototype_map[func_name] = func_prototype
                # print('>\t{}'.format(func_prototype_map.keys()))
                archs = ['origin', 'sub', 'fla']
                arch_unmangled_funcs = {}
                for arch in archs:
                    arch_unmangled_funcs[arch] = {}
                    bin_path = '{}.{}.run'.format(base, arch)
                    if os.path.isfile(bin_path):
                        funcs_folder = os.path.splitext(bin_path)[0] + '_functions'
                        if os.path.isdir(funcs_folder):
                            for f in os.listdir(funcs_folder):
                                if os.path.splitext(f)[1] != '.gdl':
                                    continue
                                fpath = os.path.join(funcs_folder, f)
                                extracted_func = os.path.splitext(f)[0]
                                if extracted_func.startswith('_'):
                                    unmangled_func = subprocess.check_output(['c++filt', '-p', '-t', extracted_func]).decode('utf-8', 'ignore').strip()
                                else:
                                    unmangled_func = extracted_func
                                '''
                                if not unmangled_func in func_prototype_map:
                                    print('>>>\tUnable to find {} in func_prototype_map for {} with arch {}'.format(unmangled_func, extracted_func, arch))
                                '''
                                '''
                                if unmangled_func in func_prototype_map:
                                    print('>>>\tFound {} mapping to function {} in source code {} with arch {}'.format(extracted_func, unmangled_func, path, arch))
                                '''
                                if unmangled_func in func_prototype_map:
                                    arch_unmangled_funcs[arch][unmangled_func] = extracted_func
                            '''
                            xor_set = set(arch_unmangled_funcs[arch].keys()) ^ set(func_prototype_map.keys())
                            if len(xor_set) > 0:
                                print('>>>\t{} does not contains: {}'.format(arch, xor_set))
                                for func in arch_unmangled_funcs[arch]:
                                    print('>>>>>\t{} -> {}'.format(func, arch_unmangled_funcs[arch][func]))
                            '''
                            for func in arch_unmangled_funcs[arch]:
                                valid_func_list_path = os.path.join(funcs_folder, 'valid_func_list.txt')
                                with open(valid_func_list_path, 'w') as list_f:
                                    list_f.write(func + ' ' + arch_unmangled_funcs[arch][func] + '\n')

                            # print('>>>\t{}-> {}'.format(arch, arch_unmangled_funcs[arch]))
        except Exception as e:
            print(path)
            print('Unexpected exception in list_function_names: {}'.format(traceback.format_exc()))



def main(argv):
    parser = argparse.ArgumentParser(description='Write function names which appear in both the source code and binary. (Can only deal with Google Code Jam dataset.)')
    parser.add_argument('TargetFolder', help='The parent folder of each author folder.')
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    q = manager.Queue()
    lock = manager.Lock()
    processes = []

    num_process = 5
    for i in range(num_process):
        p = multiprocessing.Process(target=list_function_names, args=(q, lock,))
        p.start()
        processes.append(p)

    valid_exts = ['.c', '.cc', '.cpp']
    for author_name in os.listdir(args.TargetFolder):
        author_dir = os.path.join(args.TargetFolder, author_name)
        for item in os.listdir(author_dir):
            base, ext = os.path.splitext(item)
            item_path = os.path.join(author_dir, item)
            if ext in valid_exts:
                q.put(item_path)

    for proc in processes:
        proc.join()


if __name__ == '__main__':
    main(sys.argv)
