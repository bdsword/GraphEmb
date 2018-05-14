#!/usr/bin/env python3

import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Check which folder have not been analyzed by ida pro.')
parser.add_argument('TargetDir', help='The target folder to check.')
args = parser.parse_args()

valid_ida_db_ext = ['.idb', '.i64']

for author_dir in os.listdir(args.TargetDir):
    author_dir_path = os.path.join(args.TargetDir, author_dir)
    for f in os.listdir(author_dir_path):
        file_path = os.path.join(author_dir_path, f)
        items = os.path.splitext(file_path)
        base_path = items[0]
        ext = items[1]
        if ext == '.run':
            found = False
            for db_ext in valid_ida_db_ext:
                db_path = base_path + db_ext
                if os.path.isfile(db_path):
                    found = True
                    break
            if not found:
                print('{} have not been analyzed.'.format(author_dir_path))
                break

