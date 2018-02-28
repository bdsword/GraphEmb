#!/usr/bin/env python3
import glob
import os
import argparse
import subprocess


parser = argparse.ArgumentParser(description='Transfer the .gdl files in the given path into .dot files.')
parser.add_argument('TargetDir', help='A folder path to be processed.')
args = parser.parse_args()

for gdl_file in glob.glob(os.path.join(args.TargetDir, '**', '*.gdl'), recursive=True):
    dot_file = os.path.splitext(gdl_file)[0] + '.dot'
    print('>>> Processing {} ...'.format(gdl_file))
    subprocess.call(['graph-easy', '--input', gdl_file, '--output', dot_file])

