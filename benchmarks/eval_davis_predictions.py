"""
Goal: Automatic benchmarking of a model's predictions with DAVIS-type metrics (J&F, J, and F).
Quick use:
- python benchmarks/eval_davis_predictions.py
- python benchmarks/eval_davis_predictions.py --method UXMem_GT
- python benchmarks/eval_davis_predictions.py --method UXMem_GT_mem_update
- python benchmarks/eval_davis_predictions.py --method UXMem_GT_@0.5
- python benchmarks/eval_davis_predictions.py --method UXMem_GT_@0.9

by Stéphane Vujasinovic
"""

### - IMPORTS ---
import os
from pathlib import Path
from argparse import ArgumentParser
from davis_benchmark.benchmark import benchmark

# Import from parent folder
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from configuration.configuration_manager import ConfigManager
from icecream import ic
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
    parser.add_argument("--dataset", type=str, default='d17-val',
                        help='To define the dataset')
    parser.add_argument('-n', '--num_processes', default=8, type=int,
                        help='Number of concurrent processes.')
    parser.add_argument('-s', '--strict',
                        help='Make sure every video in the ground-truth has a '
                        'corresponding video in the prediction.',
                        action='store_true')
    # https://github.com/davisvideochallenge/davis2017-evaluation/blob/d34fdef71ce3cb24c1a167d860b707e575b3034c/davis2017/evaluation.py#L85
    parser.add_argument('--do_not_skip_first_and_last_frame',
                        help='By default, we skip the first and the last '
                        'frame in evaluation following DAVIS semi-supervised'
                        'evaluation.'
                        'They should not be skipped in unsupervised '
                        'evaluation.',
                        action='store_true')
    parser.add_argument("-v", "--verbose", action="store_false",
                        help="Add icecream statements")
    return parser.parse_args()


# - MAIN ---
if __name__ == '__main__':
    args = arguments_parser()
    dataset_name = args.dataset
    method_name = args.method
    if args.verbose:
        ic.disable()

    # Prep. configuration
    config = ConfigManager()
    config['dataset_name'] = dataset_name
    config['method_name'] = method_name
    config_generator = config.get_my_configuration()

    # Load the directories locations/path
    _, gt_mask_directory = next(config_generator)
    pd_mask_directory, _, _ = next(config_generator)
    benchmark_results_dir, csv_file_name = config.get_benchmark_dir_location()

    # Prepare folders and files to save results
    csv_benchmark_results = os.path.join(benchmark_results_dir, csv_file_name)
    create_directory_if_not_there(benchmark_results_dir)
    print(f'\t -> Results are save @: {csv_benchmark_results}\n')

    # Benchmark results
    benchmark([gt_mask_directory], [pd_mask_directory], csv_benchmark_results,
              args.strict, args.num_processes, verbose=True,
              skip_first_and_last=not args.do_not_skip_first_and_last_frame)
