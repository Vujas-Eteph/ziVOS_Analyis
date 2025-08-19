'''
Goal: Compute the Entropy of the results offline.
Quick use:
- python analyze_iou_vs_entropy.py
- python analyze_iou_vs_entropy.py --method UXMem_GT
- python analyze_iou_vs_entropy.py --method UXMem_GT_mem_update
- python analyze_iou_vs_entropy.py --method UXMem_GT_@0.5
- python analyze_iou_vs_entropy.py --method UXMem_GT_@0.9

by Stephane Vujasinovic
'''

# - IMPORTS ---
import os
import numpy as np
import argparse
from argparse import Namespace

from matplotlib import pyplot as plt

from icecream import ic
import tqdm
import cv2

from utils.dataframe_manipulator import (flatten_metrics_dict, flatten_dataset_data_correlation,
                                         save_dataframe_as_csv_table,
                                         extract_obx_metrics, save_dataframe_with_polars)
from utils.imgmask_operations.image_manipulator import ImageManipulator
from utils.imgmask_operations.mask_manipulator import extract_contours, filter_entropy, adjust_entropy_with_density
from utils.statistics.entropy_operations import EntropyHelper, operation_on_Entropy_over_TPTNFPFN

from configuration.configuration_manager import ConfigManager
from utils.path_utils import find_subdirectories, get_files_from_sequence, \
    discard_unpaired_data, create_directory_if_not_there

from utils.data_exporter import DataExporter

import warnings

from scipy.stats import pearsonr, spearmanr

from utils.confusions import extract_TPTNFPFN
from utils.statistics.metrics_helper import extract_metrics

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
    parser.add_argument("--mask_H", type=int, default=5,
                        help="Mask the total entropy of the frame with a mask "
                        "around the prediction of an object. Based on the "
                        "size of the object to delimite the region used in "
                        "the Entropy calculation. In %centage")
    parser.add_argument('-n', '--num_processes', default=6, type=int,
                    help='Number of concurrent processes.')
    parser.add_argument("--verbose", action="store_false",
                        help="Add icecream statements")
    parser.add_argument("--debug", action="store_true",
                        help="Display graph, only with nummber of process set to 1")
    parser.add_argument("--warning", action="store_true",
                        help="Display warnings")
    parser.add_argument("-T", "--temperature", type=float, default=1.0,
                        help="Analysze the entropy from the temperature scaling")
    parser.add_argument("--dilate", type=str, default="dilate",
                        help= "Apply a dilation operation")
    # Load/Set arguments
    return parser.parse_args()


def load_data(
    data_exp,
    dir: str,
    sequence: str
):
    try:
        return data_exp.read_data(sequence,
                                  f'{os.path.join(dir, sequence)}')
    except FileNotFoundError:
        warnings.warn(
            f"Entropy for sequence {sequence} does not exist",
            category=Warning
        )
        return None


def gen_location_for_the_csv_file(_save_analysis_loc, args):
    _save_analysis_loc = os.path.join(_save_analysis_loc, "raw_stats_HUB")
    if args.mask_H is not None:
        _save_analysis_loc = os.path.join(_save_analysis_loc, 
                                          f"mask_H_{args.mask_H}")
    else:
        _save_analysis_loc = os.path.join(_save_analysis_loc, "no_mask_H")

    return _save_analysis_loc

def get_masked_entropy(
    tp: np.ndarray,
    fx: np.ndarray,
    H_fdx: np.ndarray,
    value_for_mask_H: int,
    type_of_op='dilate',
    debug=False
) -> np.ndarray:
    
    kernel = gen_kernel(value_for_mask_H, tp.sum() + fx.sum(), type_of_op)
    # Dilate the mask based on the kernel
    Elem = tp + fx
    Elem = Elem.astype(np.uint8)
    if type_of_op == 'dilate':
        cv2_operation = cv2.dilate
    else:
        cv2_operation = cv2.erode
    dilated_mask = cv2_operation(Elem, kernel, iterations=1).astype(bool)
    H_fdx = H_fdx * dilated_mask
    if debug:
        cv2.imshow("dilates_mask", dilated_mask.astype(np.uint8) * 255)
        cv2.imshow("not_dilates_mask", Elem * 255)

        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()

    return H_fdx


def gen_kernel(
    x: int,
    object_size: int,
    type_of_op='dilate'
) -> np.ndarray:
    # using sqrt because the object growth is square based
    if type_of_op == 'dilate':
        kernel_size = int(np.sqrt((x / 100) * object_size))
    else:
        # In this case erode
        object_size_X_percent = 10
        kernel_size = int(np.sqrt(object_size_X_percent))
    if kernel_size % 2 == 1:  # if kernel is odd
        kernel_size = kernel_size + 1
    kernel_size = max(2, kernel_size)
    # Create a circle based kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    center_of_kernel = [int(kernel.shape[0] / 2),
                        int(kernel.shape[0] / 2)]
    radius = center_of_kernel[0] - 1
    points_y = np.arange(0, int(center_of_kernel[0]))
    points_x = np.arange(0, int(center_of_kernel[0]))
    points_yy, points_xx = np.meshgrid(points_y, points_x)
    points = np.stack((points_yy.flatten(),
                       points_xx.flatten()), axis=-1)
    distance = np.square(points[:, 0]) + np.square(points[:, 1])
    in_circle = distance < np.square(radius)
    one_fourth_of_the_array = in_circle.reshape(center_of_kernel[0], -1)
    kernel[center_of_kernel[0]:, center_of_kernel[0]:] = one_fourth_of_the_array
    kernel[center_of_kernel[0]:, :center_of_kernel[0]] = one_fourth_of_the_array[:, ::-1]
    kernel[:center_of_kernel[0], :] = kernel[center_of_kernel[0]:, :][::-1]

    return kernel


# - MAIN ---
def foo(sequence_data):
    (load_data, get_masked_entropy, args, op_func,
     image_directory, gt_mask_directory, pred_mask_directory,
     entropy_directory, confusion_directory,
     loc_to_save_csv_results, image_api, data_exporter,
     seq_objx_count, meta_data_sequence, sqx, sequence_name) = sequence_data
    ic(f'sdx: {sqx} | sequence: {sequence_name}')
    # if sequence_name != 'dance-twirl': return None
    # plotcrafter.set_sequence_name(sequence_name)

    # Definitions
    sequence_data = dict()  # What does it do
    total_entropy = []      # What does it do

    # List all files in the sequence
    gt_masks_file_names = \
            get_files_from_sequence(gt_mask_directory, sequence_name)
    images_file_names = \
            get_files_from_sequence(image_directory, sequence_name)

    # Check if there exists corresponding predictions for the sequence
    try:
        predictions_file_names = \
                get_files_from_sequence(pred_mask_directory, sequence_name)
    except FileNotFoundError:
        warnings.warn(
                f"Predictions for sequence {sequence_name} does not exist",
                category=Warning
            )
        return None

    # Discard unpaired GTs/PDs
    (gt_with_a_counterpart_pd,
         image_with_a_counterpart_pd,
         predictions_file_names) = \
             discard_unpaired_data(gt_masks_file_names,
                                   images_file_names,
                                   predictions_file_names)

    # Get confusion and entropy results
    # confusions_r are obtained from [compute_confusion.py]
    # entropy_r are obtained from [entropy_operations.py]
    if args.temperature != 1.0:
        entropy_directory = entropy_directory + f'_Temp_{args.temperature}'
        loc_to_save_csv_results += f'_Temp_{args.temperature}'
    # ic.enable()
    # ic(sqx, data_exporter, confusion_directory, sequence_name)
    # ic.disable()
    confusions_r = load_data(data_exporter, confusion_directory, sequence_name)
    entropy_r = load_data(data_exporter, entropy_directory, sequence_name)
    if confusions_r is None or entropy_r is None:  # skip sequence if no data
        return None

    if confusions_r.shape[1] != entropy_r.shape[0]:
        raise(f"Confusion ({len(confusions_r)}) and Entropy "
              f"Error: ({len(entropy_r)}) don't cover the same amount of frames")

    if confusions_r.shape[1] != len(predictions_file_names):
        print(len(confusions_r))
        print(len(predictions_file_names))
        print("Filter here the files... so that I don't have to rely on some indexes later on")
        raise("Error: Number of confusions is a missmatch with number of frames")


    # Read object instances to appear in the sequence
    total_nbr_of_objects = list(set(seq_objx_count[sequence_name]))

    zipped_inputs = zip(gt_with_a_counterpart_pd,
                        image_with_a_counterpart_pd,
                        predictions_file_names)
    # - Loop over images/annotations in sequence ---
    for fdx, (gt_mask_name, image_name, prediction_name) in enumerate(zipped_inputs):
        entropy_fdx = entropy_r[fdx, 0, :, :]

        # Definitions
        frame_data = dict()  # What does it do ?

        # Find @
        gt_mask_loc = os.path.join(gt_mask_directory,
                                   sequence_name,
                                   gt_mask_name)
        pd_mask_loc = os.path.join(pred_mask_directory,
                                   sequence_name,
                                   prediction_name)

        # Load information
        gt_palette = image_api.load_image_with_PIL(gt_mask_loc)
        pd_palette = image_api.load_image_with_PIL(pd_mask_loc)
        Nbr_of_objx_in_GT = np.unique(gt_palette)
        Nbr_of_objx_in_PD = np.unique(pd_palette)

        # Save meta information
        meta_data_sequence[sequence_name] = {
                'Number_of_frames': len(predictions_file_names),
                'Total_number_of_objects': total_nbr_of_objects,
                'Resolution': gt_palette.shape
                }

        # - Loop over the objects ---
        for obx in total_nbr_of_objects:
            # exclude the background by default
            if obx == 0:
                stable_entropy_frame_fdx = entropy_fdx.copy() # stable copy
                continue

            # Extract TP, FN, FP and FN regions per frame per object
            TP, TN, FP, FN = extract_TPTNFPFN(confusions_r, fdx, obx)

            # Check if object is present in GT and in PD - frame level
            Obj_in_GT_flag = obx in Nbr_of_objx_in_GT
            Obj_in_PD_flag = obx in Nbr_of_objx_in_PD

            # Handle IoU, VOTS cases, etc
            IoU, _, _ = extract_metrics(TP, TN, FP, FN,
                                        Obj_in_GT_flag,
                                        Obj_in_PD_flag)

            # Apply a mask around the object of interest to delimite the 
            # uncertainty calculation
            gt_masked_entropy_fdx = get_masked_entropy(TP, FN, 
                                                       entropy_fdx, 
                                                       args.mask_H,
                                                       args.dilate, 
                                                       args.debug)
            pd_masked_entropy_fdx = get_masked_entropy(TP, FP, 
                                                       entropy_fdx, 
                                                       args.mask_H,
                                                       args.dilate, 
                                                       args.debug)

            # Compute the entropies
            Tot_H = op_func(entropy_fdx)
            TP_H, TN_H, FP_H, FN_H = \
                    operation_on_Entropy_over_TPTNFPFN(entropy_fdx,
                                                       TP, TN, FP, FN,
                                                       op_func)
            TR_H = TP_H + TN_H
            FR_H = FP_H + FN_H

            Tot_H_maksed_gt = op_func(gt_masked_entropy_fdx)
            (TP_H_masked_gt, TN_H_masked_gt,
                 FP_H_masked_gt, FN_H_masked_gt) = \
                    operation_on_Entropy_over_TPTNFPFN(gt_masked_entropy_fdx,
                                                       TP, TN, FP, FN,
                                                       op_func)
            TR_H_masked_gt = TP_H_masked_gt + TN_H_masked_gt
            FR_H_masked_gt = FP_H_masked_gt + FN_H_masked_gt

            Tot_H_maksed_pd = op_func(pd_masked_entropy_fdx)
            (TP_H_masked_pd, TN_H_masked_pd,
                 FP_H_masked_pd, FN_H_masked_pd) = \
                    operation_on_Entropy_over_TPTNFPFN(pd_masked_entropy_fdx,
                                                       TP, TN, FP, FN,
                                                       op_func)
            TR_H_masked_pd = TP_H_masked_pd + TN_H_masked_pd
            FR_H_masked_pd = FP_H_masked_pd + FN_H_masked_pd

            # Store the results in a dict
            frame_data[f'object_{obx}'] = {
                    # Presence of the object
                    'Obj_in_GT_flag': Obj_in_GT_flag,
                    'Obj_in_PD_flag': Obj_in_PD_flag,
                    # Size of the regions
                    'TP_size': TP.sum(), 'TN_size': TN.sum(),
                    'FP_size': FP.sum(), 'FN_size': FN.sum(),
                    'TR_size': TP.sum() + TN.sum(),
                    'FR_size': FP.sum() + FN.sum(),
                    # Global sizes
                    'Image_size': TP.sum() + TN.sum() + FP.sum() + FN.sum(),
                    'gt_object_size': TP.sum() + FN.sum(),
                    'pd_object_size': TP.sum() + FP.sum(),
                    # Performance metrics
                    'IoU': IoU,
                    # Total Entropy values
                    'Total_H_base': Tot_H,
                    # Entropy for each region
                    'TP_H_base': TP_H, 'TN_H_base': TN_H,
                    'FP_H_base': FP_H, 'FN_H_base': FN_H,
                    # Entropy for True and False regions
                    'TR_H_base': TR_H, 'FR_H_base': FR_H,
                    # Same thing, but with the masked version
                    'Masked_value': args.mask_H,
                    # GT Masked result
                    'Total_H_masked_gt': Tot_H_maksed_gt,
                    'TP_H_masked_gt': TP_H_masked_gt,
                    'TN_H_masked_gt': TN_H_masked_gt,
                    'FP_H_masked_gt': FP_H_masked_gt,
                    'FN_H_masked_gt': FN_H_masked_gt,
                    'TR_H_masked_gt': TR_H_masked_gt,
                    'FR_H_masked_gt': FR_H_masked_gt,
                    # PD Masked result
                    'Total_H_masked_pd': Tot_H_maksed_pd,
                    'TP_H_masked_pd': TP_H_masked_pd,
                    'TN_H_masked_pd': TN_H_masked_pd,
                    'FP_H_masked_pd': FP_H_masked_pd,
                    'FN_H_masked_pd': FN_H_masked_pd,
                    'TR_H_masked_pd': TR_H_masked_pd,
                    'FR_H_masked_pd': FR_H_masked_pd,
                }

        #     ic(TP.sum(), TN.sum(), FP.sum(), FN.sum())
        #     ic(obx, Obj_in_GT_flag, Obj_in_PD_flag, IoU)
        #     ic(TP_H, TN_H, FP_H, FN_H)
        #
        sequence_data[f'frame_{fdx}'] = frame_data
        # total_entropy.append(Tot_H)

        # Adjust the total entropy by discarding the first frame to stay 
        # consistent with the DAVIS protocol. / 
        # Should also discard the last frame. 
        # TODO: Move this outside of this loop and discard the index
        # total_H = np.array(total_entropy)[1:]
        # Flatten the data in a table

    # - WORKING ON THE DATAFRAME ---
    data = flatten_metrics_dict(sequence_data)
    # print(data)
    # csv_file_name = os.path.join(loc_to_save_csv_results,
    #                              f"{sequence_name}.csv")
    create_directory_if_not_there(loc_to_save_csv_results)
    csv_file_name = os.path.join(loc_to_save_csv_results,
                                     f"{sequence_name}.parquet")

    # print(csv_file_name)
    save_dataframe_with_polars(data,
                                   csv_file_name)
    
    return None

if __name__ == "__main__":
    args = arguments_parser()
    dataset_name = args.dataset
    method_name = args.method

    # Useful for debugging
    if args.verbose:
        ic.disable()
    if not args.warning:
        warnings.filterwarnings("ignore", category=Warning)


    op_func = np.sum

    # # Define correlation measurement
    # if args.correlation == 'spearman':
    #     compute_correlation = spearmanr
    # elif args.correlation == 'pearson':
    #     compute_correlation = pearsonr

    # Prep. configuration
    config = ConfigManager()
    config['dataset_name'] = dataset_name
    config['method_name'] = method_name
    config_generator = config.get_my_configuration()

    # Load the directories locations
    image_directory, gt_mask_directory = next(config_generator)
    pred_mask_directory, logits_directory, softmax_directory = \
        next(config_generator)
    entropy_directory, confusion_directory = next(config_generator)
    preprocessing_directory = next(config_generator)
    save_analysis_loc = config.get_iou_entropy_analysis_path()
    loc_to_save_csv_results = gen_location_for_the_csv_file(save_analysis_loc,
                                                            args)
    create_directory_if_not_there(loc_to_save_csv_results)

    ic(image_directory, gt_mask_directory)
    ic(pred_mask_directory, logits_directory, softmax_directory)
    ic(entropy_directory, confusion_directory)
    ic(preprocessing_directory)

    # Instance initialization
    image_api = ImageManipulator()  # Visualization API
    stat_api = EntropyHelper()      # Statistics API
    data_exporter = DataExporter()  # Data Exporter API
    data_exporter.file_extension = args.ext
    # plotcrafter = PlotDetailsIoUEntropyCorr(config.get_plot_config("evaluation"))
    # plotcrafter.set_method_name(method_name)
    # plotcrafter.set_dataset_name(dataset_name)

    # List all sequences in the ground truth annotations directory
    sequence_names = find_subdirectories(gt_mask_directory)
    ic(sequence_names)

    # Get the global object_id for each sequence in the dataset
    seq_objx_count = \
        DataExporter.read_from_JSON_format(os.path.join(preprocessing_directory,
                                                        'number_of_objects.json'))

    # Future dataframes
    dataset_data = dict()
    meta_data_sequence = dict()  # not used atm

    # - Loop over all sequences ---
    # for sqx, sequence_name in enumerate(sequence_names):
    #     sequence_data = (load_data, get_masked_entropy, args, op_func,
    #                      image_directory, gt_mask_directory, pred_mask_directory,
    #                      entropy_directory, confusion_directory,
    #                      loc_to_save_csv_results, image_api, data_exporter,
    #                      seq_objx_count, meta_data_sequence, sqx, sequence_name)
    #      foo(sequence_data)
        
        
    sequence_data_list = [(load_data, get_masked_entropy, args, op_func,
                         image_directory, gt_mask_directory, pred_mask_directory,
                         entropy_directory, confusion_directory,
                         loc_to_save_csv_results, image_api, data_exporter,
                         seq_objx_count, meta_data_sequence, sqx, sequence_name) for sqx, sequence_name in enumerate(sequence_names)]

    chunksize = 1

    with Pool(processes=args.num_processes) as pool:
        # Wrapping pool.imap_unordered with tqdm to show progress
        # Specify the total number of tasks to accurately display the progress
        for _ in tqdm.tqdm(pool.imap_unordered(foo,
                                               sequence_data_list,
                                               chunksize=chunksize),
                           total=len(sequence_data_list)):
            pass
        
    print('\n - Completed the IoU vs Entropy analysis for the '
          f'{dataset_name} dataset -')

