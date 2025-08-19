"""
Compute the percentage/ratio of frames with an IoU above X threshold. 

Prerequesits: ensure that compute_confusions was already run
Quick run:
python compute_IoU_ratio.py


by Stéphane Vujasinovic
"""

# - IMPORTS ---
import os
import argparse
import numpy as np
import pandas as pd

from configuration.configuration_manager import ConfigManager
from utils.data_exporter import DataExporter
from utils.path_utils import find_subdirectories, get_files_from_sequence, \
    discard_unpaired_data, create_directory_if_not_there

import tqdm
from icecream import ic

import warnings

from multiprocessing import Pool, Manager


# - FUNCTIONS ---
def arguments_parser():
    """
    Return parser arguments
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
    # Load/Set arguments
    return parser.parse_args()


def load_data(
    data_exp,
    dir: str,
    sequence: str
):
    return data_exp.read_data(sequence, f'{os.path.join(dir, sequence)}')


def compute_ratio(sequence_data):
    (sequence_name, image_directory, gt_mask_directory, pd_mask_directory,
     data_exporter, confusion_directory, X_dict) = sequence_data
    # List all files in the sequence
    gt_masks_file_names = get_files_from_sequence(gt_mask_directory,
                                                  sequence_name)
    images_file_names = get_files_from_sequence(image_directory,
                                                sequence_name)
    
    ic.enable()
    ic(sequence_name)
    ic.disable()

    # Check if there exists corresponding predictions for the sequence
    try:
        pd_file_names = get_files_from_sequence(pd_mask_directory,
                                                sequence_name)
    except FileNotFoundError:
        return None
    # Discard unpaired GTs/PDs
    (gt_with_a_counterpart_pd,
     image_with_a_counterpart_pd,
     pd_file_names) = discard_unpaired_data(gt_masks_file_names,
                                            images_file_names,
                                            pd_file_names)

    # Know number of classes present in advance (Needed ??)
    # total_nbr_of_objects = list(set(seq_objx_count[sequence_name]))
    confusions_r = load_data(data_exporter, confusion_directory, sequence_name)

    # Filter, background class and first frame (optional: last frame also)
    confusions_r = confusions_r[:, :, 1:, :, :]
    confusions_r = confusions_r[:, 1:, :, :, :]  # discard the first frame
    confusions_r = confusions_r[:, :-1, :, :, :]  # discard the last frame

    TP = confusions_r[0, :, :, :, :].sum(axis=-1).sum(axis=-1)
    TN = confusions_r[1, :, :, :, :].sum(axis=-1).sum(axis=-1)
    FP = confusions_r[2, :, :, :, :].sum(axis=-1).sum(axis=-1)
    FN = confusions_r[3, :, :, :, :].sum(axis=-1).sum(axis=-1)
    IoU = TP/(TP+FP+FN)
    IoU = np.nan_to_num(IoU)  # convert nan values to 0
    PD = TP + TN

    # Filter the occlusion
    Z_dict = dict()
    for obx in range(0, confusions_r.shape[2]):
        IoU_obj = IoU[:, obx]
        # Occlusion mask or object not present...
        obx_occ_mask = PD[:, obx] == 0.0
        # If the method predicts an occ, set IoU to 1 as in VOTS Challenge
        IoU_obj[obx_occ_mask] = (IoU_obj[obx_occ_mask] == 0.0)*1.0
        Z_dict[obx] = IoU_obj

    # Check ratio of frames above IoU@X
    for obx, obx_iou in Z_dict.items():
        # Vectorize operation
        tile_obx_iou = np.tile(obx_iou, (IoUatX.shape[0], 1))
        tile_IoUatX = np.tile(IoUatX, (tile_obx_iou.shape[1], 1)).T
        # Count number of IoU above IoU@X and get ratio
        ratio = tile_obx_iou >= tile_IoUatX
        ratio = ratio.sum(axis=-1)/ratio.shape[-1]
        ratio = np.around(ratio, decimals=4)
        X_dict[f"{sequence_name}_{obx+1}"] = ratio

    return None

# - MAIN ---
if __name__ == "__main__":
    args = arguments_parser()
    dataset_name = args.dataset
    method_name = args.method

    # Useful for debugging
    if args.verbose:
        ic.disable()
    if not args.warning:
        warnings.filterwarnings("ignore", category=Warning)

    # Prep. configuration
    config = ConfigManager()
    config['dataset_name'] = dataset_name
    config['method_name'] = method_name
    config_generator = config.get_my_configuration()

    # Load the directories locations
    image_directory, gt_mask_directory = next(config_generator)
    pd_mask_directory, logits_directory, softmax_directory = \
        next(config_generator)
    _, confusion_directory = next(config_generator)
    preprocessing_directory = next(config_generator)

    data_exporter = DataExporter()  # Data Exporter API
    data_exporter.file_extension = args.ext

    # List all sequences in the ground truth annotations directory
    sequence_names = find_subdirectories(gt_mask_directory)
    ic(sequence_names)
    # Get the global object_id for each sequence in the dataset
    seq_objx_count = \
        DataExporter.read_from_JSON_format(os.path.join(preprocessing_directory,
                                                        'number_of_objects.json'))

    # - Loop over all sequences ---
    X_dict = dict()
    step = 0.01
    IoUatX = np.around(np.arange(0.0, 1.0 + step, step), decimals=4)
    chunksize = 1   
    with Manager() as manager:
        X_dict = manager.dict()
        sequence_data_list = [(sequence, image_directory,
                               gt_mask_directory, pd_mask_directory,
                               data_exporter, confusion_directory, X_dict) for sequence in sequence_names]

        with Pool(processes=args.num_processes) as pool:
            # Wrapping pool.imap_unordered with tqdm to show progress
            # Specify the total number of tasks to accurately display the progress
            for _ in tqdm.tqdm(pool.imap_unordered(compute_ratio,
                                                   sequence_data_list,
                                                   chunksize=chunksize),
                               total=len(sequence_data_list)):
                pass

        final_X_dict = dict(X_dict)

    # Create table
    df = pd.DataFrame(final_X_dict)
    df['mean'] = df.mean(axis=1)
    df['threshold'] = IoUatX
    ic(df)

    # Save table
    dir_to_save_results = os.path.join("raw_stats_HUB", "ratio", dataset_name)
    print(dir_to_save_results)
    create_directory_if_not_there(dir_to_save_results)
    file_loc = os.path.join(dir_to_save_results, f"{method_name}.csv")
    df.to_csv(file_loc)
