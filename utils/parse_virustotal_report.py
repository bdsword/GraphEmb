#!/usr/bin/env python3

import ast
import sys


if len(sys.argv) != 2:
    print('Usage:\n\t{} <virustotal report>'.format(sys.argv[0]))
    sys.exit(-1)


def detected_benign_rate(scan_dict):
    counter = 0
    for engine in scan_dict:
        if scan_dict[engine]['detected'] == False:
            counter += 1
    return counter / len(scan_dict.keys())


benign_rate = 0.9

f = open(sys.argv[1], 'r')
lines = f.readlines()

for line in lines:
    report = ast.literal_eval(line)
    if report['verbose_msg'] == 'The requested resource is not among the finished, queued or pending scans':
        print('{} is not scanned.'.format(report['resource']))
    elif report['verbose_msg'] == 'Scan finished, information embedded':
        if detected_benign_rate(report['scans']) >= benign_rate:
            print('{} is benign'.format(report['md5']))
        else:
            print('{} is malicious'.format(report['md5']))

f.close()
