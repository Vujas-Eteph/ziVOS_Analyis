'''
Count the total number of objects present for the sequence.
Quick use:
- python compute_count_nbr_of_objx.py
- python compute_count_nbr_of_objx.py --method UXMem_GT
- python compute_count_nbr_of_objx.py --method UXMem_GT_mem_update
- python compute_count_nbr_of_objx.py --method UXMem_GT_@0.5
- python compute_count_nbr_of_objx.py --method UXMem_GT_@0.9

Stephane Vujasinovic
'''
import numpy as np
from icecream import ic
import argparse
import tqdm
import os
from configuration.configuration_manager import ConfigManager
from utils.imgmask_operations.image_manipulator import ImageManipulator
from utils.data_exporter import DataExporter

from utils.path_utils import create_directory_if_not_there, \
    find_subdirectories, get_files_from_sequence, discard_unpaired_data

import warnings


# - FUNCTIONS ---
def arguments_parser():
    """ Argument parser
    """
    parser = argparse.ArgumentParser(
        description="Visualize the output scores(softmax)")
    parser.add_argument("--method", type=str, default='XMem',
                        help='To define the method')
    parser.add_argument("--dataset", type=str, default='d17-val',
                        help='To define the dataset')
    parser.add_argument("--ext", type=str, default='NPZ',
                        help='which extension to use')
    parser.add_argument("--verbose", action="store_false",
                        help="Add icecream statements")
    parser.add_argument("--warning", action="store_true",
                        help="Add icecream statements")
    return parser.parse_args()


# - MAIN ---
if __name__ == '__main__':
    args = arguments_parser()
    dataset_name = args.dataset
    method_name = args.method
    if args.verbose:
        ic.disable()
    if not args.warning:
        warnings.filterwarnings("ignore", category=Warning)

    # Prep. instances
    config = ConfigManager()
    image_api = ImageManipulator()
    data_exporter = DataExporter()
    data_exporter.file_extension = args.ext

    # Prep. configuration
    config['dataset_name'] = dataset_name
    config['method_name'] = method_name
    config_generator = config.get_my_configuration()

    # Load the directories locations/path
    image_directory, gt_mask_directory = next(config_generator)
    pred_mask_directory, _, _ = \
        next(config_generator)
    _, _ = next(config_generator)
    preprocessing_directory = next(config_generator)

    ic(pred_mask_directory)

    # List all sequences in the ground truth annotations directory
    sequence_names = find_subdirectories(gt_mask_directory)

    seq_objx_count = dict()
    # - Loop over all sequences ---
    for sdx, sequence in enumerate(sequence_names):
        print(sdx, sequence)

        # List all files in the sequence
        gt_masks_file_names = \
            get_files_from_sequence(gt_mask_directory, sequence)
        images_file_names = \
            get_files_from_sequence(image_directory, sequence)
        # Check if there exists corresponding predictions for the sequence
        try:
            pd_file_names = \
                get_files_from_sequence(pred_mask_directory, sequence)
        except FileNotFoundError:
            warnings.warn(
                f"Predictions for sequence {sequence} does not exist",
                category=Warning
            )
            continue

        (gt_with_a_counterpart_pd,
         image_with_a_counterpart_pd,
         pd_file_names) = discard_unpaired_data(gt_masks_file_names,
                                                images_file_names,
                                                pd_file_names)

        # Loop over GT annotations in sequence:
        zipped_inputs = zip(gt_with_a_counterpart_pd,
                            image_with_a_counterpart_pd,
                            pd_file_names)
        enumerate_inputs = enumerate(zipped_inputs)

        seen_obj_count = set()

        # - Loop over frames
        for fdx, (gt_mask_name, image_name,
                  prediction_name) in tqdm.tqdm(enumerate_inputs):
            ic(fdx)
            gt_mask = os.path.join(gt_mask_directory,
                                   sequence,
                                   gt_mask_name)
            gt = image_api.load_image_with_PIL(gt_mask)
            gt_objects = np.unique(gt)
            seen_obj_count.update(set(gt_objects))

        seq_objx_count[sequence] = [int(elem) for elem in list(seen_obj_count)]

    print(f"Results: {seq_objx_count}")
    create_directory_if_not_there(preprocessing_directory)
    filename = os.path.join(preprocessing_directory,
                            'number_of_objects.json')
    DataExporter.save_to_JSON_format(seq_objx_count, filename)
