#!/usr/bin/env python3

import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Check which folder have not been analyzed by ida pro.')
parser.add_argument('TargetDir', help='The target folder to check.')
args = parser.parse_args()

valid_ida_db_ext = ['.idb', '.i64']

print('IDA Pro analyze check...')
for year in os.listdir(args.TargetDir):
    year_dir = os.path.join(args.TargetDir, year)
    for contest in os.listdir(year_dir):
        contest_dir = os.path.join(year_dir, contest)
        for author_dir in os.listdir(contest_dir):
            author_dir_path = os.path.join(contest_dir, author_dir)
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
                        print('{} have not been analyzed.'.format(file_path))
                        continue


print('--------------------------------------')
print('Check call graph extraction...')
for year in os.listdir(args.TargetDir):
    year_dir = os.path.join(args.TargetDir, year)
    for contest in os.listdir(year_dir):
        contest_dir = os.path.join(year_dir, contest)
        for author_dir in os.listdir(contest_dir):
            author_dir_path = os.path.join(contest_dir, author_dir)
            for f in os.listdir(author_dir_path):
                file_path = os.path.join(author_dir_path, f)
                items = os.path.splitext(file_path)
                base_path = items[0]
                ext = items[1]
                if ext == '.run':
                    gdl_path = base_path + '.gdl'
                    if not os.path.isfile(gdl_path):
                        print('{} have not extracted call graph.'.format(file_path))
                        continue
                    dot_path = base_path + '.dot'
                    if not os.path.isfile(dot_path):
                        print('{} have not converted gdl to dot format.'.format(file_path))
                        continue


print('--------------------------------------')
print('Check function flow graph extraction...')
for year in os.listdir(args.TargetDir):
    year_dir = os.path.join(args.TargetDir, year)
    for contest in os.listdir(year_dir):
        contest_dir = os.path.join(year_dir, contest)
        for author_dir in os.listdir(contest_dir):
            author_dir_path = os.path.join(contest_dir, author_dir)
            for f in os.listdir(author_dir_path):
                file_path = os.path.join(author_dir_path, f)
                items = os.path.splitext(file_path)
                base_path = items[0]
                ext = items[1]
                if ext == '.run':
                    function_dir = base_path + '_functions'
                    if not os.path.isdir(function_dir):
                        print('{} have not extracted function flow graph.'.format(file_path))
                        break
                    for fn in os.listdir(function_dir):
                        function_fpath = os.path.join(function_dir, fn)
                        if os.path.splitext(function_fpath)[1] == '.gdl':
                            if not os.path.isfile(os.path.splitext(function_fpath)[0] + '.dot'):
                                print('Functions dir of {} have not converted gdl to dot format.'.format(file_path))
                                break
