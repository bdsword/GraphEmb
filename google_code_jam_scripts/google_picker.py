#!/usr/bin/env python3

from shutil import copyfile
import pickle
import zipfile
import sys
import os
import re
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('YearParentDir', help='The path to the parent folder of year folder.')
parser.add_argument('OutputRootDir', help='The path to the output root folder.')
parser.add_argument('OutputSQLite3', help='The path to the output sqlite3 db file.')
args = parser.parse_args()


TABLE_NAME = 'code_information'
conn = sqlite3.connect(args.OutputSQLite3)
cur = conn.cursor()
cur.execute('CREATE TABLE {} (year varchar(32), contest varchar(128), author varchar(256), problem int, code_name varchar(32));'.format(TABLE_NAME))
cur.execute('CREATE INDEX year ON {}(year);'.format(TABLE_NAME))
cur.execute('CREATE INDEX contest ON {}(contest);'.format(TABLE_NAME))
cur.execute('CREATE INDEX author ON {}(author);'.format(TABLE_NAME))
cur.execute('CREATE INDEX problem ON {}(problem);'.format(TABLE_NAME))
cur.execute('CREATE INDEX code_name ON {}(code_name);'.format(TABLE_NAME))
conn.commit()


for year in os.listdir(args.YearParentDir):
    year_dir = os.path.join(args.YearParentDir, year)
    output_year_dir = os.path.join(args.OutputRootDir, year)
    if not os.path.isdir(output_year_dir):
        os.mkdir(output_year_dir)
    for contest_name in os.listdir(year_dir):
        contest_dir = os.path.join(year_dir, contest_name)
        output_contest_dir = os.path.join(output_year_dir, contest_name)
        if not os.path.isdir(output_contest_dir):
            os.mkdir(output_contest_dir)
        author_question_maps = {}
        author_count = 0
        for author in os.listdir(contest_dir):
            if author_count >= 20:
                break
            author_dir = os.path.join(contest_dir, author)
            output_author_dir = os.path.join(output_contest_dir, author)
            if not os.path.isdir(output_author_dir):
                os.mkdir(output_author_dir)
            for code in os.listdir(author_dir):
                code_path = os.path.join(author_dir, code)
                output_code_path = os.path.join(output_author_dir, code)
                copyfile(code_path, output_code_path)
                problem = int(os.path.splitext(code)[0])
                cur.execute('INSERT INTO {} (year, contest, author, problem, code_name) VALUES ("{}", "{}", "{}", {}, "{}");'.format(TABLE_NAME, year, contest_name, author, problem, code))
            author_count += 1

conn.commit()
conn.close()
