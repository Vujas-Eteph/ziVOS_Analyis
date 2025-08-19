"""
Display intermediate results with OpenCV - Check/Inspect

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
import easygui

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

os.chdir(os.getcwd().rstrip("/apps"))


# - FUNCTIONS ---
def message():
    message = """
    Intermediate results visualizations.
    Keyboard Commands:
    - n: Next frame
    - b: Previous frame
    - o: New object
    - u: Previous object
    - i: Display information on the frame
    - s: Skip to next sequence
    - a: Return to previous sequence
    - q: Quit Visualization

    Legends:
    - GT: Green contours
    - TP: Yellow
    - FP: Salmon
    - FN: Blue
    """
    print(message)
    # Display the message in a pop-up window
    easygui.msgbox(message, title="Keyboard Commands")


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


def add_text_to_image_bottom(
    im_array: np.ndarray,
    text: str
):
    """Put text on image"""
    position = (10, im_array.shape[0]-10)  # (x, y) coordinates
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # BGR color (green in this case)
    font_thickness = 2
    cv2.putText(im_array, text, position,
                font, font_scale, font_color, font_thickness)


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
    parser.add_argument("--verbose", action="store_false",
                        help="Show print statements")
    parser.add_argument("--resize", action="store_false",
                        help=f"Don't resize the images width to {default_width}")
    parser.add_argument("--op", type=str, default='sum',
                        help="""Which op. to use to compute the entopy.
                                Available options are 'sum', 'mean'""")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Only show particulat sequence")
    parser.add_argument("--object", type=int, default=None,
                        help="Only look for a particular object idx")
    parser.add_argument("--start", type=int, default=None,
                        help="Start from this frame")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Store the videos instead of showing them")
    parser.add_argument("-FX", "--FX", action="store_true",
                        help="Is set to true, combine the FP,FN together, "
                        "as well as the TP and T together")
    parser.add_argument("--size", action="store_true",
                        help="devide bythe object's size")
    parser.add_argument("--masked", action="store_true",
                        help="Apply a mask around the entropy based on the PD")
    parser.add_argument("--Son", action="store_true",
                        help="Entropy on! Show it to me baby")
    parser.add_argument("-C", "--contour", action="store_true",
                        help="Use contours")

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

# Constants
NEW_WIDTH = 480
ALPHA = 0.5
GREEN = np.array([0, 250, 0], dtype=np.uint8)
BLUE = np.array([250, 0, 0], dtype=np.uint8)
SALMON = np.array([148, 95, 234], dtype=np.uint8)
YELLOW = np.array([0, 210, 250], dtype=np.uint8)
# Variables
breaking_algo = False
VideoWriter_initialized = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 5
COLOR_MAP = cv2.COLORMAP_VIRIDIS
COLOR_MAP = cv2.COLORMAP_MAGMA
print_on_entropy_info_on_the_frame = False

# - MAIN ---
if __name__ == "__main__":
    args = arguments_parser(NEW_WIDTH)
    if not args.save:
        message()
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
    print(pred_mask_directory)
    entropy_directory, confusion_directory = next(config_generator)
    preprocessing_directory = next(config_generator)

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
        sequence = sequence_names[sdx]
        if args.sequence is not None:
            if sequence != args.sequence:
                sdx += 1
                continue
            args.sequence = None

        print(f'sdx: {sdx} | sequence name: {sequence}')

        # List all files in the sequence
        gt_masks_file_names = \
            get_files_from_sequence(gt_mask_directory, sequence)
        images_file_names = \
            get_files_from_sequence(image_directory, sequence)

        # Check if there exists corresponding predictions for the sequence
        try:
            predictions_file_names = \
                get_files_from_sequence(pred_mask_directory, sequence)
        except FileNotFoundError:
            warnings.warn(
                f"Predictions for sequence {sequence} does not exist",
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

        # Get confusion and entropy results
        # confusions_r are obtained from [compute_confusion.py]
        # entropy_r are obtained from [entropy_operations.py]
        confusions_r = load_data(data_exporter, confusion_directory, sequence)
        if confusions_r is None:  # no data, skip sequence
            sdx += 1
            continue

        if args.Son:
            entropy_r = load_data(data_exporter, entropy_directory, sequence)
            if entropy_r is None:  # no data, skip sequence
                sdx += 1
                continue

        # Read object instances to appear in the sequence
        total_nbr_of_objects = list(set(seq_objx_count[sequence]))

        zipped_inputs = zip(gt_with_a_counterpart_pd,
                            image_with_a_counterpart_pd,
                            predictions_file_names)
        enumerate_inputs = enumerate(zipped_inputs)
        # - Loop over the objects in the frame---
        obx = 1  # skip background
        while obx < len(total_nbr_of_objects):
            if args.object is not None:
                if obx != args.object:
                    obx += 1
                    continue
                args.object = None

            # - Loop over images/annotations in sequence ---
            fdx = 0
            next_sequence = False
            while fdx < len(gt_with_a_counterpart_pd):
                if args.start is not None:
                    if fdx < args.start:
                        fdx += 1
                        continue
                    args.start = None

                gt_mask_name = gt_with_a_counterpart_pd[fdx]
                image_name = image_with_a_counterpart_pd[fdx]
                prediction_name = predictions_file_names[fdx]
                if args.Son:
                    entropy_frame_fdx = entropy_r[fdx, 0, :, :]

                # Find @
                im_loc = os.path.join(image_directory, sequence, image_name)
                gt_mask_loc = os.path.join(gt_mask_directory,
                                           sequence,
                                           gt_mask_name)
                pd_mask_loc = os.path.join(pred_mask_directory,
                                           sequence,
                                           prediction_name)

                # Load informations
                im = image_api.load_image_with_OpenCV(im_loc)
                gt_palette = image_api.load_image_with_PIL(gt_mask_loc)
                pd_palette = image_api.load_image_with_PIL(pd_mask_loc)
                Nbr_of_objx_in_GT = np.unique(gt_palette)
                Nbr_of_objx_in_PD = np.unique(pd_palette)

                # Check if object is present in GT and in PD - frame level
                Obj_in_GT_flag = obx in Nbr_of_objx_in_GT
                Obj_in_PD_flag = obx in Nbr_of_objx_in_PD

                # Extract TP, FN, FP and FN regions per frame per object
                TP, TN, FP, FN = extract_TPTNFPFN(confusions_r, fdx, obx)

                # Handle IoU, VOTS cases, etc
                IoU, VOTS_success, VOTS_case = extract_metrics(TP, TN, FP, FN,
                                                               Obj_in_GT_flag,
                                                               Obj_in_PD_flag)
                if args.masked:
                    Elem = TP+FP
                    object_size = TP.sum() + FP.sum()  # Based on PD size
                    Elem = Elem.astype(np.uint8)
                    object_size_X_percent = (5/100)*object_size  #5% of object size
                    # using sqrt because the object growth is square based
                    kernel_size = int(np.sqrt(object_size_X_percent))
                    if kernel_size % 2 == 1:  # if kernel is odd
                        kernel_size = kernel_size + 1
                    kernel_size = max(2, kernel_size)

                    # Create a circle based kernel
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    center_of_kernel = [int(kernel.shape[0]/2),
                                        int(kernel.shape[0]/2)]
                    radius = center_of_kernel[0]-1

                    points_y = np.arange(0, int(center_of_kernel[0]))
                    points_x = np.arange(0, int(center_of_kernel[0]))
                    points_yy, points_xx = np.meshgrid(points_y, points_x)

                    points = np.stack((points_yy.flatten(),
                                       points_xx.flatten()), axis=-1)

                    distance = np.square(points[:, 0]) + np.square(points[:, 1])
                    in_circle = distance < np.square(radius)
                    one_fourth_of_the_array = \
                        in_circle.reshape(center_of_kernel[0], -1)
                    kernel[center_of_kernel[0]:, center_of_kernel[0]:] = \
                        one_fourth_of_the_array
                    kernel[center_of_kernel[0]:, :center_of_kernel[0]] = \
                        one_fourth_of_the_array[:, ::-1]
                    kernel[:center_of_kernel[0], :] = \
                        kernel[center_of_kernel[0]:, :][::-1]
                    # apply the mask on

                    dilated_mask = \
                        cv2.dilate(Elem, kernel, iterations=1).astype(bool)
                    entropy_frame_fdx = None
                    if args.Son:
                        entropy_frame_fdx = entropy_frame_fdx * dilated_mask

                TP_H = TN_H = FP_H = FN_H = TX_H = FX_H = None
                if args.Son:
                    TP_H, TN_H, FP_H, FN_H = \
                        operation_on_Entropy_over_TPTNFPFN(entropy_frame_fdx,
                                                        TP, TN, FP, FN, op_func)
                    if args.size:
                        object_size = TP.sum() + FP.sum()  # Based on PD size
                        TP_H = TP_H / np.array(object_size)
                        TN_H = TN_H / np.array(object_size)
                        FP_H = FP_H / np.array(object_size)
                        FN_H = FN_H / np.array(object_size)

                    if args.FX:
                        TX_H = TP_H + TN_H
                        FX_H = FP_H + FN_H

                    Total_H = TP_H + TN_H + FP_H + FN_H

                # Color the image with the masks (alpha value)
                im = add_mask_to_im(im, TP, YELLOW)
                im = add_mask_to_im(im, FP, SALMON)
                im = add_mask_to_im(im, FN, BLUE)

                # Color the image with the contours
                if args.contour:
                    gt_for_obj = gt_palette == obx
                    gt_contours = image_api.get_contours(gt_for_obj, width=5)
                    TP_contours = image_api.get_contours(TP, width=5)
                    FP_contours = image_api.get_contours(FP, width=5)
                    FN_contours = image_api.get_contours(FN, width=5)
                    im = add_contours_to_im(im, TP_contours, YELLOW)
                    im = add_contours_to_im(im, FP_contours, SALMON)
                    im = add_contours_to_im(im, FN_contours, BLUE)
                    im = add_contours_to_im(im, gt_contours, GREEN)

                if args.resize:
                    im = resize(im, NEW_WIDTH)
                display=im
                
                if args.Son:
                    # Adapt efficient entropy for OpenCV - from float64 to uint8
                    H_gray = (entropy_frame_fdx*255).astype(np.uint8)
                    H_viridis = cv2.applyColorMap(H_gray, COLOR_MAP)
                    if args.masked:
                        mask_V_contours = image_api.get_contours(dilated_mask,
                                                                    width=5)
                        masked_H_viridis = add_contours_to_im(H_viridis.copy(),
                                                            mask_V_contours,
                                                            GREEN)
                    else:
                        masked_H_viridis = H_viridis.copy()
                    # Resize image along the width dimension:
                    if args.resize:
                        H_viridis = resize(H_viridis, NEW_WIDTH)
                        masked_H_viridis = resize(masked_H_viridis, NEW_WIDTH)

                    # concatenate along the width dimension
                    display = np.concatenate((display,
                                            H_viridis,
                                            masked_H_viridis), axis=1)
                    display_padding = \
                        np.zeros([40, *display.shape[1:]], dtype=np.uint8)
                    display = np.concatenate((display_padding,
                                            display,
                                            display_padding),
                                            axis=0)
                    
                # Messages
                info_dict = {}
                if args.FX:
                    if args.Son:
                        info_dict = {"Sequence": [sequence],
                                    "Frame idx": [fdx],
                                    "Object id": [obx],
                                    "Jaccard index": [float(IoU)],
                                    "Total Entropy": [float(entropy_frame_fdx.sum())],
                                    "TX - True X Entropy": [float(TX_H)],
                                    "FX - False X Entropy": [float(FX_H)],
                                    }
                    else:
                        info_dict = {"Sequence": [sequence],
                                    "Frame idx": [fdx],
                                    "Object id": [obx],
                                    }
                        
                else:
                    if args.Son:
                        info_dict = {"Sequence": [sequence],
                                    "Frame idx": [fdx],
                                    "Object id": [obx],
                                    "Jaccard index": [float(IoU)],
                                    "Total Entropy": [float(entropy_frame_fdx.sum())],
                                    "TP - True Positive Entropy": [float(TP_H)],
                                    "TF - True Negative Entropy": [float(TN_H)],
                                    "FP - False Positive Entropy": [float(FP_H)],
                                    "FN - False Negative Entropy": [float(FN_H)]
                                    }
                    else:
                        info_dict = {"Sequence": [sequence],
                                    "Frame idx": [fdx],
                                    "Object id": [obx],
                                    }

                df = pd.DataFrame(info_dict)
                message = (f"Sequence {sdx}: {sequence} - "
                           f"Frame: {fdx}/{len(gt_with_a_counterpart_pd)-1} - "
                           f"Object: {obx}/{len(total_nbr_of_objects)-1}")
                add_text_to_image_top(display, message)
                if args.Son and print_on_entropy_info_on_the_frame:
                    if args.FX:
                        messag_bis = (f"IoU: {round(float(IoU),2)} - "
                                    f"H: {round(Total_H, 2)} - "
                                    f"TX H: {round(float(TX_H),2)} - "
                                    f"FX H: {round(float(FX_H),2)}")
                        add_text_to_image_bottom(display, messag_bis)
                    else:
                        add_text_to_image_bottom(display, "Enable -FX to see the data")

                # - Save the video ---
                if args.save:
                    if not VideoWriter_initialized:
                        video_name = f'{sequence}_obj_{obx}.avi'
                        video = cv2.VideoWriter(video_name,
                                                fourcc,
                                                fps,
                                                (display.shape[1],
                                                 display.shape[0]))
                        VideoWriter_initialized = True

                    video.write(display)
                    fdx += 1
                    continue

                else:
                    cv2.imshow("Image | Total Uncertainty | Masked", display)
                    key = cv2.waitKey(1)

                if ord("i") == key:  # Add print informations
                    print("\n------------------------------------------------")
                    print(df.to_markdown())
                    if print_on_entropy_info_on_the_frame:
                        print_on_entropy_info_on_the_frame = False
                    else:
                        print_on_entropy_info_on_the_frame = True
                    print("------------------------------------------------")
                elif ord("n") == key:
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
                    next_sequence = True
                    break
                elif ord("a") == key:
                    next_sequence = True
                    sdx -= 3
                    break
                elif ord("q") == key:
                    breaking_algo = True
                    break

            if args.save:
                video.release()
                breaking_algo = True                
                VideoWriter_initialized = False
                break

            obx += 1

            if next_sequence:
                break

            if breaking_algo:
                break

        sdx += 1

        if breaking_algo:
            break

    # Close the image window
    cv2.destroyAllWindows()
