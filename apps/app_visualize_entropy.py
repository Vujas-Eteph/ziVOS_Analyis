"""
Only to save video , temporary script

Quick use (and prefered way):
python app_opencv.py --size --FX --masked

Concrete example:
python results_inspector.py --sequence gold-fish --object 3 --start 15

by Stéphane Vujasinovic
"""

# - IMPORTS ---
import os
import argparse
import numpy as np
import warnings
import pandas as pd
import cv2

from tabulate import tabulate

import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from configuration.configuration_manager import ConfigManager
from utils.imgmask_operations.image_manipulator import ImageManipulator
from utils.data_exporter import DataExporter

from utils.path_utils import find_subdirectories, get_files_from_sequence, \
    discard_unpaired_data
from utils.confusions import extract_TPTNFPFN
from utils.statistics.metrics_helper import extract_metrics
from utils.statistics.entropy_operations import operation_on_Entropy_over_TPTNFPFN




# - CONSTANS ---
NEW_WIDTH = 600
ALPHA = 0.5
COLOR_MAP = cv2.COLORMAP_MAGMA
# Variables
breaking_algo = False
VideoWriter_initialized = False
# Use this format for web based videos
fourcc = cv2.VideoWriter_fourcc('V','P','8','0')


# - FUNCTIONS ---
def arguments_parser(default_width):
    """
    Return parser arguments
    """
    parser = argparse.ArgumentParser(
        description="Visualize the output scores(softmax)"
    )
    parser.add_argument("--method", type=str, default='XMem',
                        help='To define the method')
    parser.add_argument("--dataset", type=str, default='d17-val',
                        help='To define the dataset')
    parser.add_argument("--ext", type=str, default='NPZ',
                        help='which extension to use')
    parser.add_argument("--resize", action="store_false",
                        help=f"Don't resize the images width to {default_width}")
    parser.add_argument("--op", type=str, default='sum',
                        help="""Which op. to use to compute the entopy.
                                Available options are 'sum', 'mean'""")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Only show particulat sequence")
    parser.add_argument("--start", type=int, default=None,
                        help="Start from this frame")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Store the videos instead of showing them")
    parser.add_argument("--fps", type=int, default=20,
                        help="FPS to use")
    parser.add_argument("--verbose", action="store_false",
                        help="Show print/icecream statements")

    return parser.parse_args()

def add_text_to_image_top(
    im_array: np.ndarray,
    text: str
):
    """Put text on image"""
    position = (10, 30)  # (x, y) coordinates
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # BGR color (green in this case)
    font_thickness = 2
    cv2.putText(im_array, text, position,
                font, font_scale, font_color, font_thickness)


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


def add_contours_to_im(
    im: np.ndarray,
    contours: np.ndarray,
    color: np.ndarray
) -> np.ndarray:
    # Color the image with the contours
    im_copy = im.copy()
    contours_ref = np.transpose(contours, (1, 2, 0)).squeeze()
    contours_RGB = np.stack((contours_ref,
                             contours_ref,
                             contours_ref), axis=-1)
    gt_contours_RGB_colored = contours_RGB * color

    for rgbdx in range(0, im.shape[-1]):
        im_copy[:, :, rgbdx][contours_ref] = \
            gt_contours_RGB_colored[:, :, rgbdx][contours_ref]

    return im_copy


def add_mask_to_im(
    array: np.ndarray,
    bin_mask: np.ndarray,
    color: np.ndarray,
    alpha=0.5
) -> np.ndarray:
    # Color the image with the contours
    im_copy = array.copy()
    im_copy[bin_mask] = alpha*color + (1-alpha)*im_copy[bin_mask]
    return im_copy


def resize(
    array: np.ndarray,
    new_width: int
) -> np.ndarray:
    apect_ratio = array.shape[1]/array.shape[0]  # width/height
    new_height = int(new_width/apect_ratio)
    im_resized = cv2.resize(array, dsize=(new_width, new_height),
                            interpolation=cv2.INTER_CUBIC)

    return im_resized


import skimage


def get_contours(mask: np.ndarray, width=3):
    """
    Convert mask [R,G,B] mask (DAVIS style) to a boolean mask with only the contours of the mask.
    """
    mask = mask.astype(bool)
    zero_mask = np.zeros([*mask.shape], dtype=bool)
    erosion_mask = skimage.morphology.binary_erosion(mask, footprint=np.ones((width, width)))
    contour = erosion_mask ^ mask # XOR operation
    zero_mask = np.expand_dims(contour, axis=0)

    return zero_mask  # Not filled with zeros anymore


from PIL import Image
from icecream import ic
# - MAIN ---
if __name__ == "__main__":
    args = arguments_parser(NEW_WIDTH)
    dataset_name = args.dataset
    method_name = args.method

    # Define type of operation, used in to condensed the H information
    if args.op == 'sum':
        op_func = np.sum
    elif args.op == 'mean':
        op_func = np.mean
    else:
        raise ValueError("Unsupported operation. Use 'sum' or 'mean'.")

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
    
    softmax_per_object_directory = "/home/user_id/WORK_Station/VOS/UVOS/UXMem_Undercover/Trash_test_IoUat0.0_Hat1000.0_working_False_deep_False_/d17-val/softmax_per_obj"

    # Instance initialization
    image_api = ImageManipulator()
    data_exporter = DataExporter()
    data_exporter.file_extension = args.ext

    # List all sequences in the ground truth annotations directory
    sequence_names = find_subdirectories(gt_mask_directory)

    # Get the global object_id for each sequence in the dataset
    obj_count_filename = os.path.join(preprocessing_directory,
                                      'number_of_objects.json')
    seq_objx_count = DataExporter.read_from_JSON_format(obj_count_filename)

    # - Loop over all sequences ---
    sdx = 0
    while sdx < len(sequence_names):
        sequence_name = sequence_names[sdx]
        if args.sequence is not None:
            if sequence_name != args.sequence:
                sdx += 1
                continue
            args.sequence = None

        print(f'sdx: {sdx} | sequence name: {sequence_name}')

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
            sdx += 1
            continue

        # Discard unpaired GTs/PDs
        (gt_with_a_counterpart_pd,
         image_with_a_counterpart_pd,
         predictions_file_names) = \
             discard_unpaired_data(gt_masks_file_names,
                                   images_file_names,
                                   predictions_file_names)

        # Get entropy results
        entropy_r = load_data(data_exporter, entropy_directory, sequence_name)

        # Read object instances to appear in the sequence
        total_nbr_of_objects = list(set(seq_objx_count[sequence_name]))

        zipped_inputs = zip(gt_with_a_counterpart_pd,
                            image_with_a_counterpart_pd,
                            predictions_file_names)
        enumerate_inputs = enumerate(zipped_inputs)
        
        # Load the npy
        npy_path = os.path.join(softmax_per_object_directory, sequence_name) + '.npy'
        npy_array = np.load(npy_path)
        
        # - Loop over images/annotations in sequence ---
        fdx = 0
        next_sequence = False
        while fdx < len(gt_with_a_counterpart_pd):
            gt_mask_name = gt_with_a_counterpart_pd[fdx]
            image_name = image_with_a_counterpart_pd[fdx]
            prediction_name = predictions_file_names[fdx]
            entropy_frame_fdx = entropy_r[fdx, 0, :, :]

            # Find @
            im_loc = os.path.join(image_directory, sequence_name, image_name)
            gt_mask_loc = os.path.join(gt_mask_directory,
                                       sequence_name,
                                       gt_mask_name)
            pd_mask_loc = os.path.join(pred_mask_directory,
                                       sequence_name,
                                       prediction_name)

            # Load informations
            im = image_api.load_image_with_OpenCV(im_loc)
            gt_palette = image_api.load_image_with_PIL(gt_mask_loc)
            pd_palette = image_api.load_image_with_PIL(pd_mask_loc)
            Nbr_of_objx_in_GT = np.unique(gt_palette)
            Nbr_of_objx_in_PD = np.unique(pd_palette)

            # Combine GT/PD with the image
            gt_im_cv = image_api.load_image_with_OpenCV(gt_mask_loc)
            pd_im_cv = image_api.load_image_with_OpenCV(pd_mask_loc)
            gt_back_ground_mask = gt_im_cv.sum(axis=-1) != 0
            pd_back_ground_mask = pd_im_cv.sum(axis=-1) != 0
            for channel in range(0, im.shape[-1]):
                T = ALPHA*im[:, :, channel][pd_back_ground_mask] + \
                    (1-ALPHA)*pd_im_cv[:, :, channel][pd_back_ground_mask]
                im[:, :, channel][pd_back_ground_mask] = T.astype(np.uint8)

            # - Loop over the objects in the frame---
            Soft_output = list()
            for obx in total_nbr_of_objects:
                if obx == 0:
                    continue
                # Check if object is present in GT and in PD - frame level
                Obj_in_GT_flag = obx in Nbr_of_objx_in_GT
                Obj_in_PD_flag = obx in Nbr_of_objx_in_PD

                # Combine GT/PD with the image
                gt_for_obj = gt_palette == obx
                gt_contours = np.squeeze(get_contours(gt_for_obj, width=5))

                # Apply alpha mask
                for channel in range(0, im.shape[-1]):
                    im[:, :, channel][gt_contours] =\
                        30+gt_im_cv[:, :, channel][gt_contours]
                        
                # Extract the softmax output
                softmax_output = npy_array[fdx, 0, obx-1, :, :]
                softmax_output = -1*(softmax_output*np.log(softmax_output) + (1-softmax_output)*np.log(1-softmax_output))/np.log(2)  # actually the efficient entropy... but did not change the name yet
                Soft_output.append(softmax_output)

            # Adapt efficient entropy for OpenCV - from float64 to uint8
            H_gray = (entropy_frame_fdx*255).astype(np.uint8)
            H_viridis = cv2.applyColorMap(H_gray, COLOR_MAP)
            # Resize image along the width dimension:
            if args.resize:
                im = resize(im, NEW_WIDTH)
                H_viridis = resize(H_viridis, NEW_WIDTH)

            # concatenate along the width dimension
            display = np.concatenate((im, H_viridis), axis=1)
            display_padding = \
                np.zeros([40, *display.shape[1:]], dtype=np.uint8)
            display = np.concatenate((display_padding,
                                      display,
                                      display_padding),
                                     axis=0)

            # Messages
            message = (f"Sequence {sdx}: {sequence_name} - "
                       f"Frame: {fdx}/{len(gt_with_a_counterpart_pd)-1}")
            add_text_to_image_top(display, message)

            # - Save the video ---
            if args.save:
                if not VideoWriter_initialized:
                    video_name = f'{sequence_name}.webm'
                    video = cv2.VideoWriter(video_name,
                                            fourcc,
                                            args.fps,
                                            (display.shape[1],
                                                display.shape[0]))
                    VideoWriter_initialized = True

                video.write(display)
                fdx += 1
                continue

            else:
                cv2.imshow("Image | Total Uncertainty", display)

                display_softmax = np.concatenate(Soft_output, axis=1)
                display_softmax = (display_softmax*255).astype(np.uint8)
                display_softmax = cv2.applyColorMap(display_softmax, COLOR_MAP)
                if args.resize:
                    display_softmax = resize(display_softmax, len(Soft_output)*NEW_WIDTH)

                # cv2.imshow("Softmax", display_softmax)
                # key = cv2.waitKey(1)

                # Compute the "aleatoric uncertainty"
                sum_soft = 0
                sum_soft += sum(Soft_output)
                aleatoric = (1/len(Soft_output))*(sum_soft)
                display_softmax = (aleatoric*255).astype(np.uint8)
                display_softmax = cv2.applyColorMap(display_softmax, COLOR_MAP)
                if args.resize:
                    display_softmax = resize(display_softmax, NEW_WIDTH)
                cv2.imshow("aleatoric", display_softmax)

                Mutual_info = (np.round(entropy_frame_fdx, 2) - np.round(aleatoric, 2))
                # Mutual_info[Mutual_info < 0.1] = 0.0
                display_softmax = (Mutual_info*255).astype(np.uint8)
                display_softmax = cv2.applyColorMap(display_softmax, COLOR_MAP)
                if args.resize:
                    display_softmax = resize(display_softmax, NEW_WIDTH)
                cv2.imshow("MI", display_softmax)

                key = cv2.waitKey(1)

            if ord("n") == key:
                fdx += 1
                continue
            elif ord("b") == key:
                fdx -= 1
                continue
            elif ord("o") == key:
                break
            elif ord("u") == key:
                obx -= 2
                break
            elif ord("s") == key:
                break
            elif ord("a") == key:
                sdx -= 3
                break
            elif ord("q") == key:
                breaking_algo = True
                break

        if args.save:
            video.release()
            VideoWriter_initialized = False

        if breaking_algo:
            break

        sdx += 1


# Close the image window
cv2.destroyAllWindows()
