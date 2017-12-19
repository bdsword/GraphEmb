#!/usr/bin/env python3

import sys
import os
import zipfile
import uuid
import re
import glob

def main(argv):
    src_folder = argv[1]
    output_folder = argv[2]
    if os.path.isdir(src_folder) == False:
        print('The target path is not a folder.')
        sys.exit(-1)
    if os.path.isdir(output_folder) == False:
        print('The output path is not a folder.')
        sys.exit(-2)

    # List all files in the src_folder
    for file_path in glob.iglob(src_folder + '**', recursive=True):
        # Continue if the file is not a file
        if not os.path.isfile(file_path):
            continue
        
        # Try catch to remove non-zip files
        try:
            with open(file_path, 'rb') as fh:
                zip_file = zipfile.ZipFile(fh)
                hash_name = str(uuid.uuid4())
                folder_path = os.path.join(output_folder, hash_name)
                os.mkdir(folder_path)
                found_cpp = False
                for name in zip_file.namelist():
                    pattern = re.compile('\w+\.cpp')
                    if pattern.search(name):
                        found_cpp = True
                        zip_file.extract(name, folder_path)
                        os.rename(os.path.join(folder_path, name), os.path.join(folder_path, hash_name) + '.cpp')
                if not found_cpp:
                    os.rmdir(os.path.join(output_folder, hash_name))

        except zipfile.BadZipFile as e:
            os.remove(file_path)

if __name__ == '__main__':
    main(sys.argv)
