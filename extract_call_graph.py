#!/usr/bin/env python3

import os
import glob
import sys
import re
import subprocess


idaw_path = "C:\\Program Files (x86)\\IDA 6.8\\idaw.exe"
idc_path = "Z:\\ShareFolder\\Test\\call_graph.idc"


def main(argv):
    if len(argv) != 2:
        print('Usage:\n\textract_call_grapy.py <target folder>')
        sys.exit(-1)

    files = glob.glob('{}/**/*'.format(argv[1]), recursive=True)
    for f in files:
        try:
            output = subprocess.check_output([idaw_path, '-c', '-A', "-S{}".format(idc_path), f], shell=True)
        except Exception as e:
            print('### Failed to process: {}...'.format(f))
            print(e)


if __name__ == '__main__':
    main(sys.argv)
