#!/usr/bin/env python3
import networkx as nx
import argparse
import glob
import sys
import pickle
import re
import os
import time
import progressbar
from utils.graph_utils import read_graph
from utils.graph_utils import create_acfg_from_file


def main(argv):
    parser = argparse.ArgumentParser(description='Create ACFG for each dot file under _function directory.')
    parser.add_argument('RootDir', help='A root directory to process.')
    args = parser.parse_args()

    if not os.path.isdir(args.RootDir):
        print('{} is not a valid folder.'.format(args.RootDir))
        sys.exit(-1)

    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    counter = 0
    # Parse each file name pattern to extract arch, binary name(problem id)
    for dirpath, dirnames, filenames in os.walk(args.RootDir):
        for filename in filenames:
            if dirpath.endswith('_functions') and filename.endswith('.dot'):
                dot = os.path.join(dirpath, filename)
                if os.stat(dot).st_size == 0:
                    os.remove(dot)
                    continue

                arch = 'x86_64_O0'
                try:
                    acfg = create_acfg_from_file(dot, arch)
                    path_without_ext = os.path.splitext(dot)[0]
                    acfg_path = path_without_ext + '.acfg.plk'
                    with open(acfg_path, 'wb') as f:
                        pickle.dump(acfg, f)
                except:
                    print('!!! Failed to process {}. !!!'.format(dot))
                    continue
                counter += 1
                bar.update(counter)


if __name__ == '__main__':
    main(sys.argv)

