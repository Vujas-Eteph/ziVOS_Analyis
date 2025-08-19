'''
Goal: Compute the standard deviation of the prediction. 
See if the deviation is also inversely correlated to the performance.

python compute_entropy_for_ensemble.py --method XMem_Ensemble_2 -n 1 --verbose

python compute_entropy_for_ensemble.py --method XMem_Ensemble_2

python apps/app_opencv.py --method XMem_Ensemble_2


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
    parser.add_argument("--method", type=str, default='Ensemble_v3',
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

    return parser.parse_args()


def compute_entropy(sequence_data):
    (sqx, sequence, image_directory, gt_mask_directory, pred_mask_directory,
     softmax_directory, data_exporter) = sequence_data
    ic.enable()
    ic(sqx, sequence)
    ic.disable()
    #if sequence != "7K7WVzGG": return None
    if sqx >= 6: return None
    
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
    results_directory = softmax_directory.replace("softmax", "")
    all_softmax_outputs = [elem for elem in os.listdir(results_directory) if "softmax" in elem]
    ic(sorted(all_softmax_outputs))
    ic(entropy_directory)
    all_entropy_outputs = [entropy_directory] + [entropy_directory + elem.replace("softmax","") for elem in all_softmax_outputs if "model" in elem]
    ic(all_entropy_outputs)
    
    all_softmax_outputs = all_softmax_outputs
    all_entropy_outputs = all_entropy_outputs
    total_number_of_models = len(all_entropy_outputs)
    for edx, (softmax_dir, entropy_dir) in enumerate(zip(all_softmax_outputs, all_entropy_outputs)):
        # ic.enable()
        softmax_directory = os.path.join(results_directory, softmax_dir)
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
            softmax = os.path.join(softmax_directory, sequence, softmax_name)
            soft_pd = hkl.load(softmax)            
            # Save softmax result
            if edx == 0:
                if fdx == 0:
                    mean_softmax = np.zeros([len(groundtruth_file_names),
                                                    *soft_pd.shape])
                    variance_softmax = np.zeros([len(groundtruth_file_names),
                                                 *soft_pd.shape])
                # Ensure that the vectors are of the samge size(same number of objects)
                if soft_pd.shape[0] > mean_softmax.shape[1]:
                    # add a vector to the mean_softmax tensor
                    too_add = np.zeros([len(groundtruth_file_names), 1, *mean_softmax.shape[2:]])
                    mean_softmax = np.append(mean_softmax, too_add, axis=1)
                    variance_softmax = np.append(variance_softmax, too_add, axis=1)
                mean_softmax[fdx, :, :, :] = soft_pd
            else:
                # Ensure that the vectors are of the same size (same number of objects)
                if soft_pd.shape[0] < variance_softmax.shape[1]:
                    # add a vector to the mean_softmax tensor
                    too_add = np.zeros([1, *soft_pd.shape[1:]])
                    soft_pd = np.append(soft_pd, too_add, axis=0)
                variance_softmax[fdx, :, :, :] += (mean_softmax[fdx, :, :, :] - soft_pd)**2
        
    variance_softmax /= (total_number_of_models-1)
        
    var_directory = entropy_directory.replace("Entropy", "variance")
    
    create_directory_if_not_there(var_directory)
    dir_to_save_var = os.path.join(var_directory, f'{sequence}')
    data_exporter.save_data(variance_softmax, sequence,
                            dir_to_save_var)

    ic(f'Saving variance results for sequence {sequence}')

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
