#!/usr/bin/env python3

import sys
import os
import zipfile
import uuid
import re


def main(argv):
    src_folder = argv[1]
    output_folder = argv[2]
    if os.path.isdir(src_folder) == False:
        print('The target path is not a folder.')
        sys.exit(-1)
    if os.path.isdir(output_folder) == False:
        print('The output path is not a folder.')
        sys.exit(-2)

    author_max_try = {}

    if not src_folder.endswith('/'):
        src_folder += '/'
    # List all files in the src_folder
    for name in os.listdir(src_folder):
        file_path = os.path.join(src_folder, name)
        # Continue if the file is not a file
        if not os.path.isfile(file_path):
            continue
        
        # Try catch to remove non-zip files
        try:
            with open(file_path, 'rb') as fh:
                zip_file = zipfile.ZipFile(fh)
                items = re.findall(r'(.*)_(\d+)_(\d+)', os.path.splitext(os.path.basename(file_path))[0])[0]
                author = items[0]
                problem_id = int(items[1])
                author_dir = os.path.join(output_folder, author)

                if not os.path.isfile(author_dir) and not os.path.isdir(author_dir):
                    os.mkdir(author_dir)
                name = zip_file.namelist()[0]
                items = os.path.splitext(name)
                code_ext = items[1]

                zip_file.extract(name, author_dir)
                os.rename(os.path.join(author_dir, name),
                          os.path.join(author_dir, str(problem_id) + code_ext))

        except zipfile.BadZipFile as e:
            print('{} is not a valid zip file.'.format(file_path))
            print('Maybe you should run the google_codejam_cleaner.py first.')
            sys.exit(-2)

if __name__ == '__main__':
    main(sys.argv)
