#!/usr/bin/env python3

import pickle
import networkx as nx
from utils.graph_utils import read_graph, extract_main_graph
import argparse
import os
import sys


def main(argv):
    parser = argparse.ArgumentParser(description='Create learning data for malware classification according to the given malware and benign samples.')
    parser.add_argument('--ACFG_List', help='Path to the list file of ACFGs.',)
    parser.add_argument('--DOT_List', help='Path to the list file of Dots.',)
    args = parser.parse_args()

    if not args.ACFG_List and not args.DOT_List:
        print('Should set either ACFG_List or Dot_List.')
        sys.exit(-1)
    elif args.ACFG_List and args.DOT_List:
        print('Should not set both ACFG_List and Dot_List.')
        sys.exit(-2)

    if args.ACFG_List:
        with open(args.ACFG_List, 'r') as f:
            files = [line.strip() for line in f.readlines()]
            for fpath in files:
                with open(fpath, 'rb') as fb:
                    acfg = pickle.load(fb)
                    print('{},{}'.format(fpath, len(acfg)))
    elif args.DOT_List:
        with open(args.DOT_List, 'r') as f:
            files = [line.strip() for line in f.readlines()]
            for fpath in files:
                main_cg = extract_main_graph(fpath)

                print('{},{}'.format(fpath, len(main_cg)))


if __name__ == '__main__':
    main(sys.argv)

