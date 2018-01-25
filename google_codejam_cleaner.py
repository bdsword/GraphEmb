#!/usr/bin/env python3

import sys
import os
import zipfile
import re


def soft_remove(file_path, recycle_bin):
    basename = os.path.basename(file_path)
    os.rename(file_path, os.path.join(recycle_bin, basename))
    return


def main(argv):
    src_folder = argv[1]
    output_folder = argv[2]
    recycle_bin = argv[3]
    if os.path.isdir(src_folder) == False:
        print('The target path is not a folder.')
        sys.exit(-1)
    if os.path.isdir(output_folder) == False:
        print('The output path is not a folder.')
        sys.exit(-2)
    if os.path.isdir(recycle_bin) == False:
        print('The recycle bin is not a valid folder.')
        sys.exit(-3)

    author_max_try = {}
    # List all files in the src_folder
    for f in os.listdir(src_folder):
        file_path = os.path.join(src_folder, f)
        # Continue if the file is not a file
        if not os.path.isfile(file_path):
            continue

        items = re.findall(r'(.*)_(\d+)_(\d+)', os.path.splitext(os.path.basename(file_path))[0])[0]
        author = items[0]
        problem_id = int(items[1])
        try_id = int(items[2])
        author_dir = os.path.join(output_folder, author)

        if author not in author_max_try:
            author_max_try[author] = {}
        
        if problem_id not in author_max_try[author]:
            author_max_try[author][problem_id] = -1

        if try_id > author_max_try[author][problem_id]:
            author_max_try[author][problem_id] = try_id
    
    garbage_zips = []
    for author in author_max_try:
        for problem_id in author_max_try[author]:
            for i in range(author_max_try[author][problem_id]):
                rm_zip = '{}_{}_{}.zip'.format(author, problem_id, i)
                rm_zip = os.path.join(src_folder, rm_zip)
                if os.path.isfile(rm_zip):
                    garbage_zips.append(rm_zip)

    for garbage in garbage_zips:
        soft_remove(garbage, recycle_bin)

    invalid_zips = []
    valid_exts = ['.cpp', '.c', '.cc']
    for f in os.listdir(src_folder):
        file_path = os.path.join(src_folder, f)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, 'rb') as fh:
                zip_file = zipfile.ZipFile(fh)
                namelist = zip_file.namelist()
                if len(namelist) != 1:
                    invalid_zips.append(file_path)
                    continue
                codename = namelist[0]
                ext = os.path.splitext(codename)[1]
                if ext not in valid_exts:
                    invalid_zips.append(file_path)
        except zipfile.BadZipFile as e:
            invalid_zips.append(file_path)
    for invalid in invalid_zips:
        soft_remove(invalid, recycle_bin)
if __name__ == '__main__':
    main(sys.argv)
