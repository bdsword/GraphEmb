#!/usr/bin/env python3
import os
import argparse
import subprocess
import fnmatch


parser = argparse.ArgumentParser(description='Transfer the .gdl files in the given path into .dot files.')
parser.add_argument('TargetDir', help='A folder path to be processed.')
args = parser.parse_args()

for root, dirnames, filenames in os.walk(args.TargetDir):
    for filename in fnmatch.filter(filenames, '*.gdl'):
        full_path = os.path.join(root, filename)
        dot_file = os.path.splitext(full_path)[0] + '.dot'
        print('>>> Processing {} ...'.format(full_path))
        subprocess.call(['graph-easy', '--input', full_path, '--output', dot_file])

