#!/usr/bin/env python3
import networkx as nx
import argparse
import glob
import sys
from statistical_features import statistical_features
from structural_features import structural_features
import pickle
import re
import os
import time
import progressbar


def get_statistical_features(graph, arch):
    features = dict()
    for feature_func in statistical_features:
        features[feature_func.__name__] = dict()
        for node in graph.nodes:
            code = graph.nodes[node]['label']
            features[feature_func.__name__][node] = float(feature_func(code, arch))
    return features


def get_structural_features(graph):
    features = dict()
    for feature_func in structural_features:
        features[feature_func.__name__] = feature_func(graph)
    return features 


def create_acfg_from_file(file_path, arch):
    graph = nx.drawing.nx_pydot.read_dot(file_path)

    features_dict = get_statistical_features(graph, arch)
    for feature_name in features_dict:
        nx.set_node_attributes(graph, features_dict[feature_name], feature_name)

    feature_dict = get_structural_features(graph)
    for feature_name in feature_dict:
        nx.set_node_attributes(graph, feature_dict[feature_name], feature_name)

    return graph


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

