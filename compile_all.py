#!/usr/bin/env python3

import sys
import os
import subprocess
import shutil
from config import *

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
        author_dir = os.path.join(src_folder, file_name)

        for code_name in os.listdir(author_dir):
            code_path = os.path.join(author_dir, code_name)

            for arch in archs:
                output_file = '{}.{}.exe'.format(code_path, arch)
                # Only compile when the binary does not exist
                if not os.path.isfile(output_file):
                    compile_cmd = compile_cmds[arch].format(output=output_file, src=code_path)
                    ret = subprocess.call(compile_cmd, shell=True)
                    if ret != 0:
                        print('!!!!! Failed to exec: {}\n Return code: {}\n'.format(compile_cmd, ret))
                        continue
    

if __name__ == '__main__':
    main(sys.argv)
