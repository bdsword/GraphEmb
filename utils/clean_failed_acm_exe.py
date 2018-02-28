#!/usr/bin/env python3

import sys
import re
import argparse
import os


archs = ['x86_64_O0', 'x86_64_O1', 'x86_64_O2', 'x86_64_O3', 'arm', 'win']
valid_exts = ['.c', '.cpp', '.cc']


parser = argparse.ArgumentParser(description='List the incomplete exe files.')
parser.add_argument('dir', metavar='directory', type=str, help="Parent folder of each authors' folder.")

args = parser.parse_args()

author_dirs = os.listdir(args.dir)
for author_dir in author_dirs:
    author_dir_path = os.path.join(args.dir, author_dir)
    files = os.listdir(author_dir_path)
    found_exe = False
    for f in files:
        items = os.path.splitext(f)
        base = items[0]
        ext = items[1]
        if ext == '.exe':
            found_exe = True
            break
    
    if found_exe == False:
        print(author_dir_path)
        '''
        if ext in valid_exts:
            exe_names = ['{}.{}.exe'.format(base, arch) for arch in archs]
            exist_file_paths = []
            for exe_name in exe_names:
                exe_path = os.path.join(author_dir_path, exe_name)
                if os.path.isfile(exe_path):
                    exist_file_paths.append(exe_path)
            if len(exist_file_paths) != len(archs):
                print(os.path.join(author_dir_path, f))
        '''

