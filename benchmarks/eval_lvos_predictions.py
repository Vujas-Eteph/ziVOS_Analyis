#!/usr/bin/env python
"""
Goal: Automatic benchmarking of a model's predictions for the LVOS dataset (validation and training ONLY).
Quick use: python eval_lvos_predictions.py

Comments:
    - Need circa 52 minutess with 1 thread.
    - Need circa 5 minutes with 8 thread.  

by Stéphane Vujasinovic
"""


### - IMPORTS ---
import argparse
from argparse import ArgumentParser
import os
import sys
from ast import arg
from time import time
from icecream import ic

import numpy as np
import pandas as pd

from lvos_benchmark.evaluation import LVOSEvaluation as LVOSEvaluation_SP
from lvos_benchmark.evaluation_mp import LVOSEvaluation as LVOSEvaluation_MP

# Import from parent folder
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from configuration.configuration_manager import ConfigManager
from utils.path_utils import create_directory_if_not_there


# - FUNCTIONS ---
def arguments_parser():
    """
    Argument parser

    Returns:
        _type_: _description_
    """
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="XMem")
    parser.add_argument("--dataset", type=str, default='lvos-val',
                        help='Only lvos type datasets')
    parser.add_argument('-n', '--num_processes', default=8, type=int,
                        help='Number of concurrent processes.')
    parser.add_argument('--task', type=str, default='semi-supervised',
                        help='Type of task to be performed (e.g., "semi-supervised")')
    parser.add_argument('-s', '--strict',
                        help='Make sure every video in the ground-truth has a '
                        'corresponding video in the prediction.',
                        action='store_true')
    parser.add_argument("-v", "--verbose", action="store_false",
                        help="Add icecream statements")
    return parser.parse_args()


def find_data_split(dataset_name):
    if dataset_name.split('-')[-1] == 'val':
        subset = 'valid'
    elif dataset_name.split('-')[-1] == 'train':
        subset = 'train'
    return subset


def lvos_eval_method(num_processes):
    if num_processes <= 1:
        LVOSEvaluation = LVOSEvaluation_SP
        print('Evaluating using 1 process.')
    else:
        LVOSEvaluation = LVOSEvaluation_MP
        print(f'Evaluation using {num_processes} processes.')
    return LVOSEvaluation


def lvos_evaluation(args, dataset_name, subset, gt_mask_directory, pd_mask_directory):
    lvos_eval = lvos_eval_method(args.num_processes)
    lvos_root = gt_mask_directory.rstrip("Annotations")
    if args.num_processes == 1:
        dataset_eval = lvos_eval(
                lvos_root=lvos_root,
                task=args.task,
                gt_set=subset)
    else:
        dataset_eval = lvos_eval(
                lvos_root=lvos_root,
                task=args.task,
                gt_set=subset,
                mp_procs=args.num_processes)
    metrics_res, metrics_res_seen, metrics_res_unseen = dataset_eval.evaluate(pd_mask_directory)
    J, F ,V = metrics_res['J'], metrics_res['F'], metrics_res['V']
    J_seen, F_seen, V_seen = metrics_res_seen['J'], metrics_res_seen['F'], metrics_res_seen['V']
    J_unseen, F_unseen, V_unseen = metrics_res_unseen['J'], metrics_res_unseen['F'], metrics_res_unseen['V']
     # Generate dataframe for the general results
    g_measures = ['J&F-Mean','J-Mean', 'J-seen-Mean', 'J-unseen-Mean', 'F-Mean','F-seen-Mean', 'F-unseen-Mean', 'V-Mean',  'V-seen-Mean',  'V-unseen-Mean']
    #final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    final_mean = ((np.mean(J_seen["M"]) + np.mean(F_seen["M"])) + (np.mean(J_unseen["M"]) + np.mean(F_unseen["M"])))/ 4.
    g_res = np.array([final_mean, (np.mean(J_seen["M"])+np.mean(J_unseen["M"]))/2, np.mean(J_seen["M"]), np.mean(J_unseen["M"]), (np.mean(F_seen["M"])+np.mean(F_unseen["M"]))/2, np.mean(F_seen["M"]),
                      np.mean(F_unseen["M"]), (np.mean(V_seen["M"])+np.mean(V_unseen["M"]))/2, np.mean(V_seen["M"]), np.mean(V_unseen["M"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean', 'V-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    V_per_object = [V['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object, V_per_object)), columns=seq_measures)
    return table_g, table_seq


def lvos_benchmark(args, dataset_name, subset, gt_mask_directory, pd_mask_directory, csv_name_global_path, csv_name_per_sequence_path):
    if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
        print('Read precomputed results...')
        table_g = pd.read_csv(csv_name_global_path)
        table_seq = pd.read_csv(csv_name_per_sequence_path)

        return table_g, table_seq

    print(f'Evaluating sequences for the {args.task} task...')
    table_g, table_seq = lvos_evaluation(args, dataset_name, subset, gt_mask_directory, pd_mask_directory)
    ic(csv_name_global_path)
    with open(f"{csv_name_global_path}.csv", 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")

    return table_g, table_seq


# - MAIN ---
if True:  # debugging
# if __name__ == 'main':
    args = arguments_parser()
    dataset_name = args.dataset
    method_name = args.method
    if args.verbose:
        ic.disable()

    subset = find_data_split(dataset_name)

    # Prep. configuration
    config = ConfigManager()
    config['dataset_name'] = dataset_name
    config['method_name'] = method_name
    config_generator = config.get_my_configuration()

    # Load the directories locations/path
    _, gt_mask_directory = next(config_generator)
    pd_mask_directory, _, _ = next(config_generator)
    benchmark_results_dir, csv_file_name = config.get_benchmark_dir_location()
    
    print(benchmark_results_dir)

    # Prepare folders and files to save results
    csv_benchmark_results = os.path.join(benchmark_results_dir, csv_file_name)
    print(csv_benchmark_results)
    create_directory_if_not_there(benchmark_results_dir)
    print(f'\t -> Results are save @: {csv_benchmark_results}\n')

    # - BENCHMARKING ---

    csv_name_global = f'global_results_{subset}.csv'  # Might add both of the results in one table ?? Look how DAVIS does it
    csv_name_per_sequence = f'per_sequence_results_{subset}.csv'

    # Check if the method has been evaluated before, if so read the results.
    # Otherwise compute the results
    csv_name_global_path = os.path.join(benchmark_results_dir, dataset_name)
    csv_file_name = 'TRASH_TEST.csv'

    # Check if the folder exists
    if not os.path.exists(csv_name_global_path):
        # If it doesn't exist, create it
        os.makedirs(csv_name_global_path)

    csv_benchmark_results = os.path.join(csv_name_global_path, csv_file_name)
    print(f'\t -> Results are save @: {csv_name_global_path}\n')

    # Check if the method has been evaluated before, if so read the results, otherwise compute the results
    csv_name_per_sequence_path = os.path.join(csv_name_global_path, f'{dataset_name}_per_sequence')
    csv_file_name_per_sequence = 'TRASH_TEST_per_name.csv'

    # Check if the folder exists
    if not os.path.exists(csv_name_per_sequence_path):
        # If it doesn't exist, create it
        os.makedirs(csv_name_per_sequence_path)

    csv_name_per_sequence_path = os.path.join(csv_name_per_sequence_path, csv_file_name_per_sequence)
    print(f'\t -> Results are saved @: {csv_name_per_sequence_path}\n')

    table_g, table_seq = lvos_benchmark(args, dataset_name, subset, gt_mask_directory, pd_mask_directory, csv_name_global_path, csv_name_per_sequence_path)  

    print(f'\t -> Global results are saved @: {csv_name_global_path}\n')
    print(f'\t -> Per-sequence results saved are saved @: {csv_name_per_sequence_path}\n')

    # Print the results
    print(table_g.to_string(index=False))
    print(table_seq.to_string(index=False))
