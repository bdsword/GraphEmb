#!/usr/bin/env python3

import os
import sys
import pydotplus
import glob

def main(argv):
    if len(argv) != 2:
        print('Usage:\n\tclean_single_node_graph.py <target folder>')
        sys.exit(-1)

    if not os.path.isdir(argv[1]):
        print('Target folder is not a valid folder path.')
        sys.exit(-2)
    
    target_folder = argv[1]

    for file_path in glob.iglob(os.path.join(target_folder, '**', '*.dot'), recursive=True):
        if not os.path.isfile(file_path):
            continue
        dot_graph = pydotplus.parser.parse_dot_data(open(file_path, 'r').read())
        dot_nodes = dot_graph.get_nodes()
        dot_edges = dot_graph.get_edges()

        valid_count = 0
        for node in dot_nodes:
            if node.get_label() is None:
                continue
            if len(node.get_label()) > 0:
                valid_count += 1

        if valid_count == 1:
            print('>>> Removing {}'.format(file_path))
            os.remove(file_path)


if __name__ == '__main__':
    main(sys.argv)
