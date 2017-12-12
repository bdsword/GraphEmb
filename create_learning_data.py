#!/usr/bin/env python3

import os
import sys
import numpy as np
import pickle
import re
from random import randrange
from utils import _start_shell


def main(argv):
    if len(argv) != 3:
        print('Usage:\n\tcreate_learning_data.py <acfg dictionary plk> <output learning plk>')
        sys.exit(-1)

    with open(argv[1], 'rb') as f: 
        acfg_dict = pickle.load(f)

    num_positive = 2000 
    num_negative = 2000 
    num_test = 2000

    learning_data = {'train': {'sample': [], 'label': []}, 'test': {'sample': [], 'label': []}}
    used_data = set()

    counter = 0
    # Select sample from different arch
    while(counter < num_positive):

        # Select one function from the first arch

        # Arch -> Function Name -> {FileName, Graph}
        # Random choose one arch
        available_archs = list(acfg_dict.keys())
        choosed_arch = available_archs[randrange(0, len(available_archs))]
        # Random choose one function name
        available_funcs = list(acfg_dict[choosed_arch].keys())
        choosed_func = available_funcs[randrange(0, len(available_funcs))]
        # Random choose one file name
        available_files = list(acfg_dict[choosed_arch][choosed_func].keys())
        num_choice = len(available_files)
        choice_idx = randrange(0, num_choice)
        choosed_file = available_files[choice_idx]
        graph_left = acfg_dict[choosed_arch][choosed_func][choosed_file]

        # Random choose another arch 
        other_archs = available_archs
        other_archs.remove(choosed_arch)
        if len(other_archs) < 0:
            print('Unable to choose another arch...')
            sys.exit(-2)

        another_arch = other_archs[randrange(0, len(other_archs))]
        try:
            graph_right = acfg_dict[another_arch][choosed_func][choosed_file]
        except KeyError as e:
            continue
        
        data_pattern_left = '{}_{}_{}'.format(choosed_arch, choosed_func, choosed_file)
        data_pattern_right = '{}_{}_{}'.format(another_arch, choosed_func, choosed_file)
        pair_pattern_1 = '{}~{}'.format(data_pattern_left, data_pattern_right)
        pair_pattern_2 = '{}~{}'.format(data_pattern_right, data_pattern_left)
        if pair_pattern_1 not in used_data and pair_pattern_2 not in used_data:
            learning_data['train']['sample'].append([graph_left, graph_right])
            learning_data['train']['label'].append(1)
            counter += 1
            used_data.add(pair_pattern_1)
            used_data.add(pair_pattern_2)


    used_data = set()
    counter = 0
    while counter < num_negative:
        # Arch -> Function Name -> {FileName, Graph}
        # Random choose one arch
        available_archs = list(acfg_dict.keys())
        choosed_arch = available_archs[randrange(0, len(available_archs))]
        # Random choose one function name
        available_funcs = list(acfg_dict[choosed_arch].keys())
        choosed_func = available_funcs[randrange(0, len(available_funcs))]
        # Random choose one file name
        available_files = list(acfg_dict[choosed_arch][choosed_func].keys())
        num_choice = len(available_files)
        choice_idx = randrange(0, num_choice)
        choosed_file = available_files[choice_idx]
        graph_left = acfg_dict[choosed_arch][choosed_func][choosed_file]


        # Random choose one arch
        another_arch = available_archs[randrange(0, len(available_archs))]
        # Random choose one function name
        available_funcs = list(acfg_dict[choosed_arch].keys())
        another_func = available_funcs[randrange(0, len(available_funcs))]
        # Random choose one file name
        available_files = list(acfg_dict[choosed_arch][choosed_func].keys())
        num_choice = len(available_files)
        choice_idx = randrange(0, num_choice)
        another_file = available_files[choice_idx]
        graph_left = acfg_dict[choosed_arch][choosed_func][choosed_file]

        if choosed_file == another_file and choosed_func == another_func:
            continue

        data_pattern_left = '{}_{}_{}'.format(choosed_arch, choosed_func, choosed_file)
        data_pattern_right = '{}_{}_{}'.format(another_arch, another_func, another_file)
        pair_pattern_1 = '{}~{}'.format(data_pattern_left, data_pattern_right)
        pair_pattern_2 = '{}~{}'.format(data_pattern_right, data_pattern_left)
        if pair_pattern_1 not in used_data and pair_pattern_2 not in used_data:
            learning_data['train']['sample'].append([graph_left, graph_right])
            learning_data['train']['label'].append(-1)
            counter += 1
            used_data.add(pair_pattern_1)
            used_data.add(pair_pattern_2)
    
    
    used_data = set()
    counter = 0
    # Select sample from different arch
    while(counter < num_test):
        # Arch -> Function Name -> {FileName, Graph}
        # Random choose one arch
        available_archs = list(acfg_dict.keys())
        choosed_arch = available_archs[randrange(0, len(available_archs))]
        # Random choose one function name
        available_funcs = list(acfg_dict[choosed_arch].keys())
        choosed_func = available_funcs[randrange(0, len(available_funcs))]
        # Random choose one file name
        available_files = list(acfg_dict[choosed_arch][choosed_func].keys())
        num_choice = len(available_files)
        choice_idx = randrange(0, num_choice)
        choosed_file = available_files[choice_idx]
        graph_left = acfg_dict[choosed_arch][choosed_func][choosed_file]


        # Random choose one arch
        another_arch = available_archs[randrange(0, len(available_archs))]
        # Random choose one function name
        available_funcs = list(acfg_dict[choosed_arch].keys())
        another_func = available_funcs[randrange(0, len(available_funcs))]
        # Random choose one file name
        available_files = list(acfg_dict[choosed_arch][choosed_func].keys())
        num_choice = len(available_files)
        choice_idx = randrange(0, num_choice)
        another_file = available_files[choice_idx]
        graph_left = acfg_dict[choosed_arch][choosed_func][choosed_file]

        if choosed_file == another_file and choosed_func == another_func:
            if choosed_arch == another_arch:
                continue
            else:
                label = 1
        else:
            label = -1

        data_pattern_left = '{}_{}_{}'.format(choosed_arch, choosed_func, choosed_file)
        data_pattern_right = '{}_{}_{}'.format(another_arch, another_func, another_file)
        pair_pattern_1 = '{}~{}'.format(data_pattern_left, data_pattern_right)
        pair_pattern_2 = '{}~{}'.format(data_pattern_right, data_pattern_left)
        if pair_pattern_1 not in used_data and pair_pattern_2 not in used_data:
            learning_data['test']['sample'].append([graph_left, graph_right])
            learning_data['test']['label'].append(label)
            counter += 1
            used_data.add(pair_pattern_1)
            used_data.add(pair_pattern_2)

    with open(sys.argv[2], 'wb') as f:
        pickle.dump(learning_data, f)


if __name__ == '__main__':
    main(sys.argv)
