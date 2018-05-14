#!/usr/bin/env python3

import zipfile
import sys
import os
import re
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('YearParentDir', help='The path to the parent folder of year folder.')
parser.add_argument('OutputRootDir', help='The path to the output root folder.')
args = parser.parse_args()

for year in ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']:
    year_dir = os.path.join(args.YearParentDir, year)
    output_year_dir = os.path.join(args.OutputRootDir, year)
    if not os.path.isdir(output_year_dir):
        os.mkdir(output_year_dir)
    for contest_name in os.listdir(year_dir):
        contest_dir = os.path.join(year_dir, contest_name)
        output_contest_dir = os.path.join(output_year_dir, contest_name)
        if not os.path.isdir(output_contest_dir):
            os.mkdir(output_contest_dir)
        if not os.path.isdir(contest_dir):
            continue
        author_question_maps = {}
        for zip_name in os.listdir(contest_dir):
            zip_path = os.path.join(contest_dir, zip_name)
            author, question, version = re.findall(r'(.+)_(\d+)_(\d+)\.zip', zip_name)[0]

            question = int(question)
            version = int(version)
            if author not in author_question_maps:
                author_question_maps[author] = {}

            if question not in author_question_maps:
                author_question_maps[author][question] = {'max_version': -1, 'zip_path': None}

            if version > author_question_maps[author][question]['max_version']:
                author_question_maps[author][question]['max_version'] = version
                author_question_maps[author][question]['zip_path'] = zip_path

        for author in author_question_maps:
            output_author_dir = os.path.join(output_contest_dir, author)
            if not os.path.isdir(output_author_dir):
                os.mkdir(output_author_dir)
            for question in author_question_maps[author]:
                with open(author_question_maps[author][question]['zip_path'], 'rb') as fh:
                    try:
                        zip_file = zipfile.ZipFile(fh)
                    except zipfile.BadZipFile as e:
                        print('{} is not a valid zip file.'.format(author_question_maps[author][question]['zip_path']))
                        continue

                    name = zip_file.namelist()[0]
                    items = os.path.splitext(name)
                    code_ext = items[1]

                    if code_ext not in ['.c', '.cc', '.cpp', '.cxx']:
                        continue

                    try:
                        zip_file.extract(name, output_author_dir)
                    except:
                        print('{} file is a currupted zip file.'.format(author_question_maps[author][question]['zip_path']))
                        continue
                    os.rename(os.path.join(output_author_dir, name), os.path.join(output_author_dir, str(question) + code_ext))

