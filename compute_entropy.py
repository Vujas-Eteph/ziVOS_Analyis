'''
Goal: Compute the Entropy of the results offline.
Quick use:
- python compute_entropy.py -h
- python compute_entropy.py --method UXMem_GT
- python compute_entropy.py --method UXMem_GT_mem_update
- python compute_entropy.py --method UXMem_GT_@0.5
- python compute_entropy.py --method UXMem_GT_@0.9

by Stephane Vujasinovic
'''

# - IMPORTS ---
import os
import hickle as hkl
import numpy as np
import argparse

from icecream import ic
import tqdm

from utils.data_exporter import DataExporter
from utils.imgmask_operations.image_manipulator import ImageManipulator
from utils.statistics.entropy_operations import EntropyHelper

from configuration.configuration_manager import ConfigManager
from utils.path_utils import create_directory_if_not_there, \
    find_subdirectories, get_files_from_sequence, discard_unpaired_data

import warnings

from multiprocessing import Pool

from scipy.special import softmax as scipy_softmax


# - FUNCTIONS ---
def arguments_parser():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser(
        description="Visualize the output scores(softmax)")
    parser.add_argument("--method", type=str, default='XMem',
                        help='To define the method')
    parser.add_argument("--dataset", type=str, default='d17-val',
                        help='To define the dataset')
    parser.add_argument("--ext", type=str, default='NPZ',
                        help='which extension to use')
    parser.add_argument('-n', '--num_processes', default=6, type=int,
                        help='Number of concurrent processes.')
    parser.add_argument("--verbose", action="store_false",
                        help="Add icecream statements")
    parser.add_argument("--warning", action="store_true",
                        help="Add icecream statements")
    parser.add_argument("-T", "--temperature", type=float, default=1.0,
                        help="Apply temperature scaling to the softmax output")

    return parser.parse_args()


def compute_entropy(sequence_data):
    (sqx, sequence, image_directory, gt_mask_directory, pred_mask_directory,
     softmax_directory, data_exporter) = sequence_data
    ic(sqx, sequence)

    groundtruth_file_names = get_files_from_sequence(gt_mask_directory,
                                                     sequence)
    images_file_names = get_files_from_sequence(image_directory, sequence)
    # Check if there exists corresponding predictions for the sequence
    try:
        predictions_file_names = \
            get_files_from_sequence(pred_mask_directory, sequence)
    except FileNotFoundError:
        warnings.warn(
            f"Predictions for sequence {sequence} does not exist",
            category=Warning
        )
        return None
    softmax_file_names = get_files_from_sequence(softmax_directory,
                                                 sequence)
    zip_data = zip(*discard_unpaired_data(sorted(groundtruth_file_names),
                                          sorted(images_file_names),
                                          sorted(predictions_file_names),
                                          sorted(softmax_file_names)))
    enumerate_data = enumerate(zip_data)
    # - Loop over images/annotations in sequence ---
    ic(enumerate_data)
    for fdx, (gt_mask_name, image_name, pd_mask_name, softmax_name) in enumerate_data:
        ic(gt_mask_name, image_name, pd_mask_name, softmax_name)
        softmax = os.path.join(softmax_directory, sequence, softmax_name)
        ic(softmax)
        # Compute Entropy
        if 1.0 == args.temperature:            
            try:
                soft_pd = hkl.load(softmax)
            except ValueError as e:
                warnings.warn(f"Provided argument file_obj: {softmax} is not a"
                                f" valid hickle file!. Error msg:{e},",
                                category=Warning)
                break
        else:
            #load the logits instead and apply the temperatrue
            logits = os.path.join(logits_directory, sequence, softmax_name)
            if softmax_name == '00000.hkl':
                soft_pd = hkl.load(softmax)
            else:
                logits_pd = hkl.load(logits)
                # Apply temperature scaling
                logits_pd = logits_pd/args.temperature
                soft_pd = scipy_softmax(logits_pd, axis=0)
        stat_api.entropy = soft_pd
        stat_api.norm_entropy = soft_pd
        efficient_entropy = stat_api.norm_entropy

        # Save the entropy results for each frame
        if fdx == 0:
            entropy_for_sequence = np.zeros([len(groundtruth_file_names),
                                             *efficient_entropy.shape])
        entropy_for_sequence[fdx, :, :, :] = efficient_entropy

    # Save entropy results for the sequence
    if args.temperature != 1.0:
        local_entropy_directory = entropy_directory + f'_Temp_{args.temperature}'
    else:
        local_entropy_directory = entropy_directory
    create_directory_if_not_there(local_entropy_directory)
    dir_to_save_entropy = os.path.join(local_entropy_directory, f'{sequence}')
    data_exporter.save_data(entropy_for_sequence, sequence, dir_to_save_entropy)

    ic(f'Saving entropy results for sequence {sequence}')

    return None


# - MAIN ---
if __name__ == "__main__":
    args = arguments_parser()
    dataset_name = args.dataset
    method_name = args.method
    if args.verbose:
        ic.disable()
    if not args.warning:
        warnings.filterwarnings("ignore", category=Warning)

    # Prep. configuration
    config = ConfigManager()
    config['dataset_name'] = dataset_name
    config['method_name'] = method_name
    config_generator = config.get_my_configuration()

    # Instance initialization
    image_api = ImageManipulator()
    stat_api = EntropyHelper()
    data_exporter = DataExporter()
    data_exporter.file_extension = args.ext

    # Load the directories locations/path
    image_directory, gt_mask_directory = next(config_generator)
    pred_mask_directory, logits_directory, softmax_directory = \
        next(config_generator)
    entropy_directory, _ = next(config_generator)

    # List all sequences in the ground truth annotations directory
    sequence_names = find_subdirectories(gt_mask_directory)

    # - Loop over all sequences ---
    sequence_data_list = [(sqx, sequence, image_directory,
                           gt_mask_directory, pred_mask_directory,
                           softmax_directory, data_exporter) for sqx, sequence in enumerate(sequence_names)]

    chunksize = 1

    with Pool(processes=args.num_processes) as pool:
        # Wrapping pool.imap_unordered with tqdm to show progress
        # Specify the total number of tasks to accurately display the progress
        for _ in tqdm.tqdm(pool.imap_unordered(compute_entropy,
                                               sequence_data_list,
                                               chunksize=chunksize),
                           total=len(sequence_data_list)):
            pass

    # # - Loop over all sequences ---
    # for sqx, sequence in enumerate(sequence_names):
    #     sequence_data = (sqx, sequence, image_directory, gt_mask_directory, pred_mask_directory, softmax_directory)
    #     _ = foo(sequence_data)

    print('\n - Completed the Entropy analysis for the '
          f'{dataset_name} dataset -')
