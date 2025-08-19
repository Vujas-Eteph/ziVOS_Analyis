'''
Goal: Compute the TP,TN and the FP,FN regions.
Quick use:
- python compute_confusion.py
- python compute_confusion.py --method UXMem_GT
- python compute_confusion.py --method UXMem_GT_mem_update
- python compute_confusion.py --method UXMem_GT_@0.5
- python compute_confusion.py --method UXMem_GT_@0.9

by Stephane Vujasinovic
'''

# - IMPORTS ---
import os
import numpy as np
import argparse
from icecream import ic
from matplotlib import pyplot as plt
import tqdm
#import numba

from utils.imgmask_operations.image_manipulator import ImageManipulator
from utils.imgmask_operations.mask_manipulator import compute_confusion_array, \
    adjust_TrueNegative_mask_for_missing_obx
from utils.statistics.entropy_operations import EntropyHelper
from configuration.configuration_manager import ConfigManager

from utils.path_utils import create_directory_if_not_there, \
    find_subdirectories, get_files_from_sequence, discard_unpaired_data

from utils.data_exporter import DataExporter

import warnings

from multiprocessing import Pool


# - FUNCTIONS ---
def arguments_parser():
    """
    Argument parser

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

    return parser.parse_args()

#@numba.njit
def compute_confusion(sequence_data):
    (sqx, sequence, image_directory, gt_mask_directory,
     pd_mask_directory, seq_objx_count, confusion_directory, data_exporter) = sequence_data

    # if sequence != 'paragliding-launch': return None

    # List all files in the sequence
    groundtruth_file_names = get_files_from_sequence(gt_mask_directory,
                                                     sequence)
    images_file_names = get_files_from_sequence(image_directory,
                                                sequence)
    # Check if there exists corresponding predictions for the sequence
    try:
        predictions_file_names = \
            get_files_from_sequence(pd_mask_directory, sequence)
    except FileNotFoundError:
        warnings.warn(
            f"Predictions for sequence {sequence} does not exist",
            category=Warning
        )
        return None

    total_nbr_of_objects_seen = set(seq_objx_count[sequence])
    zipped_inputs = zip(*discard_unpaired_data(groundtruth_file_names,
                                               images_file_names,
                                               predictions_file_names)
                        )
    zipped_inputs = enumerate(zipped_inputs)
    # - Loop over images/annotations in sequence ---
    for fdx, (gt_mask_name, image_name, pd_mask_name) in zipped_inputs:
        # Find @
        gt_location = os.path.join(gt_mask_directory,
                                   sequence,
                                   gt_mask_name)
        pd_location = os.path.join(pd_mask_directory,
                                   sequence,
                                   pd_mask_name)

        # Load
        gt_mask = image_api.load_image_with_PIL(gt_location)
        pd_mask = image_api.load_image_with_PIL(pd_location)

        # Get all objects in the ground-truth
        gt_objects, stack_of_gt = image_api.generate_masks_from_palette(gt_mask)
        pd_objects, stack_of_pd = image_api.generate_masks_from_palette(pd_mask)

        missing_objx_in_gt = list(set(total_nbr_of_objects_seen) - set(gt_objects))
        missing_objx_in_pd = list(set(total_nbr_of_objects_seen) - set(pd_objects))

        stack_of_GT_masks = np.zeros([len(list(total_nbr_of_objects_seen)),
                                      *stack_of_gt.shape[:2]])
        stack_of_PD_masks = np.zeros([len(list(total_nbr_of_objects_seen)),
                                      *stack_of_pd.shape[:2]])

        for _, object_idx in enumerate(gt_objects):
            stack_of_GT_masks[object_idx, :, :] = stack_of_gt[:, :, _]
        for _, object_idx in enumerate(pd_objects):
            stack_of_PD_masks[object_idx, :, :] = stack_of_pd[:, :, _]

        ic(missing_objx_in_gt, missing_objx_in_pd)
        ic(stack_of_GT_masks.shape, stack_of_PD_masks.shape)

        # image_api.generate_binary_mask_for_each_object(gt, pd)
        (TruePositive, TrueNegative,
         FalsePositive, FalseNegative) = compute_confusion_array(stack_of_GT_masks, stack_of_PD_masks)

        # # Missing object in GT and in PD respectively
        # TrueNegative = \
        #     adjust_TrueNegative_mask_for_missing_obx(missing_objx_in_gt,
        #                                              TrueNegative)
        # TrueNegative = \
        #     adjust_TrueNegative_mask_for_missing_obx(missing_objx_in_pd,
        #                                              TrueNegative)

        # Save the TP, TN, FP, FN mask results for each frame
        if fdx == 0:
            TP_reg_for_seq = np.zeros([1, len(groundtruth_file_names),
                                       *TruePositive.shape], dtype=bool)
            TN_reg_for_seq = TP_reg_for_seq.copy()
            FP_reg_for_seq = TP_reg_for_seq.copy()
            FN_reg_for_seq = TP_reg_for_seq.copy()

        TP_reg_for_seq[:, fdx, :, :, :] = TruePositive.astype(bool)
        TN_reg_for_seq[:, fdx, :, :, :] = TrueNegative.astype(bool)
        FP_reg_for_seq[:, fdx, :, :, :] = FalsePositive.astype(bool)
        FN_reg_for_seq[:, fdx, :, :, :] = FalseNegative.astype(bool)

    # Save the TP, TN, FP, FN mask results for the sequence
    confusion_array = np.concatenate([TP_reg_for_seq, TN_reg_for_seq,
                                      FP_reg_for_seq, FN_reg_for_seq],
                                     axis=0)

    create_directory_if_not_there(confusion_directory)
    dir_to_save_confusion = os.path.join(confusion_directory,
                                         f'{sequence}')
    data_exporter.save_data(confusion_array, sequence,
                            dir_to_save_confusion)

    ic(f'Saving confusion results for sequence {sequence}')

    return None


# - MAIN ---
if __name__ == '__main__':
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
    image_api = ImageManipulator()  # Visualization API
    stat_api = EntropyHelper()      # Statistics API
    data_exporter = DataExporter()
    data_exporter.file_extension = args.ext

    # Load the directories locations/path
    image_directory, gt_mask_directory = next(config_generator)
    pd_mask_directory, _, softmax_directory = \
        next(config_generator)
    _, confusion_directory = next(config_generator)
    preprocessing_directory = next(config_generator)

    # List all sequences in the ground truth annotations directory
    sequence_names = find_subdirectories(gt_mask_directory)

    # Total count of objecrs per sequence
    obj_count_filename = os.path.join(preprocessing_directory,
                                      'number_of_objects.json')
    seq_objx_count = DataExporter.read_from_JSON_format(obj_count_filename)

    # - Loop over all sequences ---
    sequence_data_list = [(sqx, sequence, image_directory,
                           gt_mask_directory, pd_mask_directory,
                           seq_objx_count, confusion_directory, data_exporter) for sqx, sequence in enumerate(sequence_names)]

    chunksize = 1

    with Pool(processes=args.num_processes) as pool:
        # Wrapping pool.imap_unordered with tqdm to show progress
        # Specify the total number of tasks to accurately display the progress
        for _ in tqdm.tqdm(pool.imap_unordered(compute_confusion,
                                               sequence_data_list,
                                               chunksize=chunksize),
                           total=len(sequence_data_list)):
            pass

    print(f'\n - Completed the Confusion analysis for the {dataset_name} dataset -')
