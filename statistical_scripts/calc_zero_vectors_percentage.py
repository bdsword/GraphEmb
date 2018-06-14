#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pickle


def main(argv):
    parser = argparse.ArgumentParser(description='Calculate the percentage of zero vectors in the call graph..')
    parser.add_argument('--SamplesDir', help='Path to the target folder that contains samples.', required=True)
    parser.add_argument('--MaxNodeNum', help='Path to the target folder that contains samples.', required=True)
    parser.add_argument('--EmbSize', help='Path to the target folder that contains samples.', required=True)
    args = parser.parse_args()

    for dirpath, dirnames, filenames in os.walk(args.SamplesDir):
        for sample in filenames:
            if sample.endswith('.maxnode{}_emb{}.acg.plk'.format(args.MaxNodeNum, args.EmbSize)):
                acg_path = os.path.join(dirpath, sample)
                with open(acg_path, 'rb') as f:
                    acg = pickle.load(f)
                    count = 0
                    for node_id in acg.nodes:
                        vec = acg.nodes[node_id]['attributes']
                        if np.allclose(vec, 0):
                            count += 1
                    print('{},{}'.format(acg_path, count / len(acg)))


if __name__ == '__main__':
    main(sys.argv)

