#!/usr/bin/env python3

import os
import sys
import r2pipe
import time
import re
import pypeg2
import subprocess
from config import *


def extract_func_names(file_path):
    key_words = ['if', 'for', 'while']
    with open(file_path, 'r') as f:
        code = f.read()
        res = re.findall(r'\w+[\s\n\t]+(\w+)[\s\n\t]*\(.*\)[\s\n\t]*\{', code)
        if len(res) > 0:
            func_names = set(res)
            for key_word in key_words:
                if key_word in func_names:
                    func_names.remove(key_word)
            return func_names 
    return None


def generate_dot(r2agent, fun_name, file_path):
    r2agent.cmd('ag {fun_name} > {binary_path}~{fun_name}.dot'.format(fun_name=fun_name, binary_path=file_path))
    return


def extract_cfg(file_path):
    r2agent = r2pipe.open(file_path)
    r2agent.cmd('aaaaa')
    funcs = {}
    for f in r2agent.cmd('afl').split('\n'):
        items = f.split()
        func_name = items[-1]
        func_addr = items[0]
        if not re.search("(sym\._)|(sym\.imp)|(sym\.mingw)|(sym\..*cxx::)|(sym\..*std::)|(sym\.register_tm_clones)|(sym\.deregister_tm_clones)", func_name) and re.search("sym\.", func_name):
            funcs[func_name] = func_addr

    for func_name in funcs:
        generate_dot(r2agent, func_name, file_path)
    return


def main(argv):
    if len(argv) != 2:
        print('Usage: compile_all.py <src folder>')
        sys.exit(-1)

    src_folder = argv[1]
    if os.path.isdir(src_folder) == False:
        print('The target path is not a folder.')
        sys.exit(-1)

    files = os.listdir(src_folder)
    for file_name in files:
        sub_folder = os.path.join(src_folder, file_name)
        src_file_path = os.path.join(sub_folder, file_name) + '.cpp'
        func_names = extract_func_names(src_file_path)
        for arch in archs:
            binary_file_path = src_file_path + '.' + arch
            if os.path.isfile(binary_file_path):
                print('>>> Processing {} ...'.format(binary_file_path))
                extract_cfg(binary_file_path)
    return


if __name__ == '__main__':
    main(sys.argv)
