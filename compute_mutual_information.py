'''
Goal: Compute mutual information.

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
    parser.add_argument('-n', '--num_processes', default=8, type=int,
                        help='Number of concurrent processes.')
    parser.add_argument("--verbose", action="store_false",
                        help="Add icecream statements")
    parser.add_argument("--warning", action="store_true",
                        help="Add icecream statements")

    return parser.parse_args()


def compute_entropy(sequence_data):
    (sqx, sequence, image_directory, gt_mask_directory, pred_mask_directory,
     softmax_directory, softmax_per_object_directory, data_exporter) = sequence_data
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
    zip_data = zip(*discard_unpaired_data(groundtruth_file_names,
                                          images_file_names,
                                          predictions_file_names,
                                          softmax_file_names))
    enumerate_data = enumerate(zip_data)
    # - Loop over images/annotations in sequence ---
    for fdx, (gt_mask_name, image_name, pd_mask_name, softmax_name) in enumerate_data:
        ic(fdx)

        # Compute Entropy
        softmax = os.path.join(softmax_directory, sequence, softmax_name)
        try:
            soft_pd = hkl.load(softmax)
        except ValueError as e:
            warnings.warn(f"Provided argument file_obj: {softmax} is not a"
                            f" valid hickle file!. Error msg:{e},",
                            category=Warning)
            break
        stat_api.entropy = soft_pd
        stat_api.norm_entropy = soft_pd
        efficient_entropy = stat_api.norm_entropy

        # Save the entropy results for each frame
        if fdx == 0:
            entropy_for_sequence = np.zeros([len(groundtruth_file_names),
                                             *efficient_entropy.shape])
        entropy_for_sequence[fdx, :, :, :] = efficient_entropy

    # Save entropy results for the sequence
    create_directory_if_not_there(entropy_directory)
    dir_to_save_entropy = os.path.join(entropy_directory, f'{sequence}')
    data_exporter.save_data(entropy_for_sequence, sequence,
                            dir_to_save_entropy)

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
    
    softmax_per_object_directory = "project/location/exp_id/dataset/softmax_per_obj"

    # List all sequences in the ground truth annotations directory
    sequence_names = find_subdirectories(gt_mask_directory)

    # - Loop over all sequences ---
    sequence_data_list = [(sqx, sequence, image_directory,
                           gt_mask_directory, pred_mask_directory,
                           softmax_directory, softmax_per_object_directory, data_exporter) for sqx, sequence in enumerate(sequence_names)]

    chunksize = 1
    args.num_processes = 1
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
