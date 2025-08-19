"""
The main application for analyzing the results.

Quick use:
- streamlit run streamlit_apps/app_entropy_performance_analysis.py 

by Stephane Vujasinovic
"""

# - IMPORTS ---
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go
import polars as pl
import pandas as pd
import numpy as np
import cv2
import os
import json
import scipy

# Import from parent folder
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from configuration.configuration_manager import ConfigManager
from utils.imgmask_operations.image_manipulator import ImageManipulator

from utils.confusions import extract_TPTNFPFN

import matplotlib.pyplot as plt


from utils.ColorBlindPalette import hex_to_rgba, \
    st_color_palette

from apps.streamlit_apps.st_utils.st_color_map import get_color_dict


from apps.streamlit_apps.st_configuration.configuration import streamlit_custom_page_configuration

# - APP APPERANCE ---
streamlit_custom_page_configuration("Entropy vs Performance Viewer", ":fire:")

pio.templates.default = "plotly_dark"
pio.templates.default = "ggplot2"
#pio.templates.default = "seaborn"

# taken from https://discuss.streamlit.io/t/issues-with-background-colour-for-buttons/38723
import streamlit.components.v1 as components
def ChangeButtonColour(widget_label):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color = 'white';
                    elements[i].style.background = 'rgba{hex_to_rgba(st_color_palette['Red'], alpha=1.0)}'; 
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)
    
def RevertButtonColour(widget_label):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.color = 'white';
                    elements[i].style.background = 'transparent'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)


# - FUNCTIONS ---
@st.cache_data
def read_from_JSON_format(
    filename: str
) -> dict:
    """Read data from a JSON file and return it as a dictionary."""
    with open(filename, 'r') as file:
        return json.load(file)

@st.cache_data
def find_subdirectories(  # Custom for streamlit
    directory: str
):
    '''
    Find the subdirectories
    '''
    try:
        return [d for d in os.listdir(directory) if os.path.isdir(
            os.path.join(directory, d))]
    except OSError as e:
        raise Exception(f"An error occurred while accessing the directory: {e}")


@st.cache_data
def get_files_from_sequence(  # Custom for streamlit
    directory_to_look_into: str,
    sequence: str
) -> str:
    return os.listdir(os.path.join(directory_to_look_into, sequence))

@st.cache_data
def filter_and_sort_filenames(
    gt_masks_file_names: list,
    file_names: list,
    extension_to_replace: str,
    target_extension='.png'
) -> list:
    return sorted([filename for filename in file_names if filename.replace(extension_to_replace, target_extension) in gt_masks_file_names])


@st.cache_data
def discard_unpaired_data(
    gt_file_names: list,
    img_file_names: list,
    pd_files_names: list,
    softmax_file_names=None
):
    """
    Discard no paired files name (DAVIS2017), and no paired GT with PD (LVOS)
    """
    # Filter GT folders that have no pairs in PD
    img_file_names_bis = \
        filter_and_sort_filenames(gt_file_names,
                                  img_file_names,
                                  '.jpg')
    pd_file_names_bis = \
        filter_and_sort_filenames(gt_file_names,
                                  pd_files_names,
                                  '.png')

    # Filter GT frames/masks that have no PD pairs
    gt_with_a_counterpart_pd = \
        filter_and_sort_filenames(pd_file_names_bis,
                                  gt_file_names,
                                  '.png')
    image_with_a_counterpart_pd = \
        filter_and_sort_filenames(pd_file_names_bis,
                                  img_file_names_bis,
                                  '.jpg')

    if softmax_file_names is not None:
        softmax_file_names = \
            filter_and_sort_filenames(gt_file_names,
                                      softmax_file_names,
                                      '.hkl')
        return (gt_with_a_counterpart_pd, image_with_a_counterpart_pd,
                pd_file_names_bis, softmax_file_names)

    return (gt_with_a_counterpart_pd, image_with_a_counterpart_pd,
            pd_file_names_bis)



def extract_performance_related_curves(
    curves_of_interest: dict
) -> dict:
    CURVES_OF_INTEREST = ["IoU"]
    new_dict = dict()
    new_dict.update({key_oI: curves_of_interest[key_oI] for key_oI in CURVES_OF_INTEREST})
    return new_dict


def extract_size_related_curves(
    curves_of_interest: dict
) -> dict:
    CURVES_OF_INTEREST = ["Image_size",
                          "TP_size", "TN_size", "FP_size", "FN_size", 
                          "TR_size", "FR_size",
                          "gt_object_size", "pd_object_size"]
    new_dict = dict()
    new_dict.update({key_oI: curves_of_interest[key_oI] for key_oI in CURVES_OF_INTEREST})
    return new_dict


def extract_presence_related_curves(
    curves_of_interest: dict
) -> dict:
    CURVES_OF_INTEREST = ["Obj_in_GT_flag",	"Obj_in_PD_flag"]
    new_dict = dict()
    new_dict.update({key_oI: curves_of_interest[key_oI] for key_oI in CURVES_OF_INTEREST})
    return new_dict


def extract_entropy_related_curves(
    curves_of_interest: dict
) -> dict:
    CURVES_OF_INTEREST = ["Total_H_base",
                          "TP_H_base", "TN_H_base",
                          "FP_H_base", "FN_H_base",
                          "TR_H_base", "FR_H_base",
                          "Total_H_masked_gt",
                          "TP_H_masked_gt", "TN_H_masked_gt",
                          "FP_H_masked_gt", "FN_H_masked_gt",
                          "TR_H_masked_gt",	"FR_H_masked_gt",
                          "Total_H_masked_pd",
                          "TP_H_masked_pd", "TN_H_masked_pd",
                          "FP_H_masked_pd", "FN_H_masked_pd",
                          "TR_H_masked_pd", "FR_H_masked_pd"]
    new_dict = dict()
    new_dict.update({key_oI: curves_of_interest[key_oI] for key_oI in CURVES_OF_INTEREST})
    return new_dict


def adapt_dataframe_with_image_size(
    curves_of_interest: dict,
) -> dict:
    EXCLUDE = ["IoU", "Image_size", "TP_size", "TN_size", "FP_size", "FN_size",
               "TR_size", "FR_size",
               "Obj_in_GT_flag", "Obj_in_PD_flag",
               "gt_object_size", "pd_object_size"]
    columns = curves_of_interest.keys()
    columns_of_interest = list(set(columns) - set(EXCLUDE))
    curves_to_div_by_objet_size = {coi: curves_of_interest[coi] for coi in columns_of_interest}
    img_sizes = curves_of_interest["Image_size"]
    curves_uncorrelated_to_object_size = {key: curves_to_div_by_objet_size[key]/img_sizes for key in curves_to_div_by_objet_size.keys()}
    curves_of_interest.update(curves_uncorrelated_to_object_size)

    return curves_of_interest

def adapt_dataframe_with_object_size(
    curves_of_interest: dict,
    _use_object_size: str
) -> dict:
    EXCLUDE = ["IoU", "Image_size", "TP_size", "TN_size", "FP_size", "FN_size",
               "TR_size", "FR_size",
               "Obj_in_GT_flag", "Obj_in_PD_flag",
               "gt_object_size", "pd_object_size"]
    columns = curves_of_interest.keys()
    columns_of_interest = list(set(columns) - set(EXCLUDE))
    curves_to_div_by_objet_size = {coi: curves_of_interest[coi] for coi in columns_of_interest}
    if _use_object_size == 'GT':
        obj_sizes = curves_of_interest["gt_object_size"]
    elif _use_object_size == 'PD':
        obj_sizes = curves_of_interest["pd_object_size"]
    else:
        st.write('not recognized')

    #check ig obj_size has 0.
    for key in curves_to_div_by_objet_size.keys():
        curves_uncorrelated_to_object_size = np.array(curves_to_div_by_objet_size[key]/obj_sizes)
        # Deal with nans and infs
        # curves_uncorrelated_to_object_size[np.isnan(curves_uncorrelated_to_object_size)] = 0
        # curves_uncorrelated_to_object_size[curves_uncorrelated_to_object_size == np.inf] = 0
        curves_of_interest.update({key: curves_uncorrelated_to_object_size})

    return curves_of_interest

def adapt_dataframe_with_mean_operation(
    curves_of_interest: dict
) -> dict:
    EXCLUDE = ["IoU", "Image_size", "TP_size", "TN_size", "FP_size", "FN_size",
               "TR_size", "FR_size",
               "Obj_in_GT_flag", "Obj_in_PD_flag",
               "gt_object_size", "pd_object_size"]
    columns = curves_of_interest.keys()
    columns_of_interest = list(set(columns) - set(EXCLUDE))
    curves_to_mean = {coi: curves_of_interest[coi] for coi in columns_of_interest}

    # Compute the mean for the TP
    prefix = "Total"
    mean_Total_curves = {key: curves_to_mean[key]/curves_of_interest["Image_size"] for key in curves_to_mean.keys() if prefix in key}

    # Compute the mean for the TP
    prefix = "TP"
    mean_TP_curves = {key: curves_to_mean[key]/curves_of_interest["TP_size"] for key in curves_to_mean.keys() if prefix in key}

    # Compute the mean for the TP
    prefix = "TN"
    mean_TN_curves = {key: curves_to_mean[key]/curves_of_interest["TN_size"] for key in curves_to_mean.keys() if prefix in key}

    # Compute the mean for the TP
    prefix = "FP"
    mean_FP_curves = {key: curves_to_mean[key]/curves_of_interest["FP_size"] for key in curves_to_mean.keys() if prefix in key}

    # Compute the mean for the TP
    prefix = "FN"
    mean_FN_curves = {key: curves_to_mean[key]/curves_of_interest["FN_size"] for key in curves_to_mean.keys() if prefix in key}

    # Compute the mean for the TR
    prefix = "TR"
    mean_TR_curves = {key: curves_to_mean[key]/curves_of_interest["TR_size"] for key in curves_to_mean.keys() if prefix in key}

    # Compute the mean for the FR
    prefix = "FR"
    mean_FR_curves = {key: curves_to_mean[key]/curves_of_interest["FR_size"] for key in curves_to_mean.keys() if prefix in key}

    # Update with the new values
    curves_of_interest.update(mean_Total_curves)
    curves_of_interest.update(mean_TP_curves)
    curves_of_interest.update(mean_TN_curves)
    curves_of_interest.update(mean_FP_curves)
    curves_of_interest.update(mean_FN_curves)
    curves_of_interest.update(mean_TR_curves)
    curves_of_interest.update(mean_FR_curves)

    return curves_of_interest


def adapt_dataframe_with_simulation(
    curves_of_interest: dict
) -> dict:
    EXCLUDE = ["IoU", "Image_size", "TP_size", "TN_size", "FP_size", "FN_size",
               "TR_size", "FR_size",
               "Obj_in_GT_flag", "Obj_in_PD_flag",
               "gt_object_size", "pd_object_size"]
    columns = curves_of_interest.keys()
    columns_of_interest = list(set(columns) - set(EXCLUDE))
    curves_to_simulate = {coi: curves_of_interest[coi] for coi in columns_of_interest}
    CURVES_OF_INTEREST = ["TP_size", "TN_size", "FP_size", "FN_size",
                          "TR_size", "FR_size"]
    curves_to_use_for_simulation = {key_oI.split('_')[0]: curves_of_interest[key_oI] for key_oI in CURVES_OF_INTEREST}
    
    for k, v in curves_to_use_for_simulation.items():
        for k_bis, _ in curves_to_simulate.items():
            if k not in k_bis:
                continue
            if "F" in k:  # False regions get an H= 1.0 for the each pixel
                curves_to_simulate[k_bis] = v
            if "T" in k:  # False regions get an H= 1.0 for the each pixel
                curves_to_simulate[k_bis] = v*0
    
    # Adapt the total amounts too
    for k_bis, _ in curves_to_simulate.items():
        if "Total" not in k_bis:
            continue
        curves_to_simulate[k_bis] = curves_to_use_for_simulation["FR"]
            
    curves_of_interest.update(curves_to_simulate)

    return curves_of_interest


def extract_curves(
    _df_obx
) -> dict:
    EXCLUDE = ["frame", "object_id", "Masked_value"]
    columns = _df_obx.columns
    columns_of_interest = list(set(columns) - set(EXCLUDE))

    curves_of_interest = {coi: _df_obx[coi] for coi in columns_of_interest}

    return curves_of_interest


@st.cache_data
def read_from_NPZ_format(
    filename: str
) -> np.ndarray:
    """Read data from an NPZ file."""
    with np.load(filename) as data:
        return next(iter(data.values()))


@st.cache_data
def load_data_confusion(
    folder: str,
    sequence: str,
):
    filename = f"{os.path.join(folder, sequence)}.npz"
    return read_from_NPZ_format(filename)


@st.cache_data
def load_data_entropy(
    folder: str,
    sequence: str
):
    filename = f"{os.path.join(folder, sequence)}.npz"
    return read_from_NPZ_format(filename)


@st.cache_resource
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


@st.cache_resource
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


@st.cache_resource
def resize(
    array: np.ndarray,
    new_width: int
) -> np.ndarray:
    apect_ratio = array.shape[1]/array.shape[0]  # width/height
    new_height = int(new_width/apect_ratio)
    im_resized = cv2.resize(array, dsize=(new_width, new_height),
                            interpolation=cv2.INTER_CUBIC)

    return im_resized


def draw_span_plotly(_fig, x, y, color, y_min=0, y_max=1):
    start = None
    for i, value in enumerate(y):
        if value == 0 and start is None:
            start = x[i] - 0.5
        elif value == 1 and start is not None:
            # Adjust end to handle isolated "1"s
            end = x[i] if y[i-1] == 1 else x[i-1] + 0.5
            # Add shape for the block
            _fig.add_shape(type="rect",
                           x0=start, y0=y_min, x1=end, y1=y_max,
                           fillcolor=color, opacity=0.2,
                           layer="below", line_width=0)
            start = None

    # Handle the case where the last value is 1
    if start is not None:
        _fig.add_shape(type="rect",
                       x0=start, y0=y_min, x1=x[-1], y1=y_max,
                       fillcolor=color, opacity=0.2,
                       layer="below", line_width=0)

# Please choose from: blue, green, orange, red, violet, gray, grey, rainbow.
st_color = 'red'

# - MAIN ---
def derive(entropy_curve):
    function_decaler = entropy_curve[1:]
    function_decaler = np.append(function_decaler,
                                 np.nan)
    derivative = function_decaler - entropy_curve
    derivative[0] = np.nan
    return derivative

if __name__ == '__main__':
    # Configuration/ Available methods and datasets
    config = ConfigManager()
    avail_datasets = config.get_all_available_datasets()
    avail_methods = config.get_all_available_methods()
    
    # Constants
    GREEN = np.array([0, 250, 0], dtype=np.uint8)
    BLUE = np.array([250, 0, 0], dtype=np.uint8)
    SALMON = np.array([181, 52, 205], dtype=np.uint8)
    YELLOW = np.array([0, 210, 250], dtype=np.uint8)

    NEW_WIDTH = 1000
    METHOD_OPTIONS = avail_methods
    DATASET_OPTIONS = avail_datasets
    OPERATION_OPTIONS = ["sum", "mean"]
    COLOR_MAP_OPTIONS = ["magma", "viridis", "cividis"]
    CORR_COEFF_OPTIONS = ["spearman", "pearson"]
    color_dict = get_color_dict()

    # - Streamlit widgets ---
    st.title(f":{st_color}[iVOTS - Interactive Visualization]")
    with st.sidebar:
        st.header(f":{st_color}[CONTROLE FLOW]", divider=st_color)
        method_name = st.selectbox("Method", METHOD_OPTIONS, key="method")
        dataset_name = st.selectbox("Dataset", DATASET_OPTIONS, key="dataset")

    # Prep. configuration
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
    parquet_file_location = os.path.join(save_analysis_loc,
                                         "raw_stats_HUB",
                                         "mask_H_5")

    # Deal with Prompts
    prompt_history = config.get_prompts_history_dir_location()
    print('hi')
    print(prompt_history)

    # Instance initialization
    image_api = ImageManipulator()

    # List all sequences in the ground truth annotations directory
    sequence_names = find_subdirectories(gt_mask_directory)

    valid_sequence_names = []
    for _ in sequence_names:
        # Check if there exists corresponding predictions for the sequence
        try:
            predictions_file_names = \
                get_files_from_sequence(pred_mask_directory, _)
            valid_sequence_names.append(_)
        except FileNotFoundError:
            continue

    with st.sidebar:
        sequence_name = st.selectbox("Sequence (sqx)",
                                     valid_sequence_names)

    # Get the global object_id for each sequence in the dataset
    seq_objx_count = read_from_JSON_format(os.path.join(preprocessing_directory,
                                                        'number_of_objects.json'))

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
        raise f"Predictions for sequence {sequence_name} does not exist"

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
    confusions_r = load_data_confusion(confusion_directory, sequence_name)
    entropy_r = load_data_entropy(entropy_directory, sequence_name)
    if confusions_r is None or entropy_r is None:  # no data, skip sequence
        raise "confusions_r or entropy_r does not exist"

    # - Loop over the objects in the frame ---
    total_nbr_of_objects = list(set(seq_objx_count[sequence_name]))
    with st.sidebar:
        skip_backgrnd = st.toggle('Skip background', True)
        if skip_backgrnd:
            total_nbr_of_objects = total_nbr_of_objects[1:]
        obx = st.selectbox("Object index (obx)", total_nbr_of_objects)

    # Filter for the sequence and the object
    if prompt_history is not None:
        prompts_for_sequence = os.path.join(prompt_history, sequence_name)
        meta_prompts = os.path.join(prompt_history, "meta_data.parquet")
        meta_prompts_df = pl.read_parquet(meta_prompts)
        date_prompts_for_obx = os.path.join(prompts_for_sequence, f"id_{obx}.parquet")
        # Read prompts into a dataframe:
        prompts_df = pl.read_parquet(date_prompts_for_obx)
        prompts_columns = prompts_df.columns
        # Extract relevant data for the prompts
        fdx_prompts = prompts_df[prompts_columns[1]]        # fdx
        prompt_calls = prompts_df[prompts_columns[0]]       # prompt
        entropy_prompts = prompts_df[prompts_columns[2]]    # Recorded Entropy
        # Check if available
        try:
            iou_prompts = prompts_df[prompts_columns[3]]    # Recorded IoU
        except:
            iou_prompts = None
        meta_data_entropy_seuil_for_prompt_call = meta_prompts_df.columns
        prompt_ent = meta_prompts_df["Threshold"].item()
        prompt_iou = meta_prompts_df["IoU@"].item()
        prompt_deriv = meta_prompts_df["Derivative"].item()

    with st.sidebar:
        st.write("#")
        st.header(f":{st_color}[EVALUATION OPTIONS]", divider=st_color)
        with st.form("eval_form"):
            col_sub_0, col_sub_1 = st.columns(2)
            with col_sub_0:
                use_sim = st.toggle('Simulation', False)
            with col_sub_1:
                use_derivatives = st.toggle('Derivatives', True)
            col1, col2 = st.columns(2)
            with col1:
                operation_type = st.selectbox("Operation",
                                              OPERATION_OPTIONS,
                                              key="op")
            with col2:
                corr_type = st.selectbox("Correlation",
                                         CORR_COEFF_OPTIONS,
                                         key="corr")
            # if operation_type
            if operation_type == 'mean':
                # If already mean, then the values are uncoupled from the region size...
                use_object_size = False
            else:
                use_object_size = st.selectbox('Decouple results from object size',
                                                options=['PD', 'GT', False],
                                                key="use_object_size")
            if operation_type == 'mean':
                use_image_size = False
            else:
                use_image_size = st.toggle('Decouple results from image size', False)

            submitted = st.form_submit_button("Submit")

        shifting_by = st.slider('shift Entropy and IoU when computing correlation',
                        min_value=-5,
                        max_value=5,
                        value=0)

        # - Visualization Options -
        st.write("#")
        st.header(f":{st_color}[VISUALIZATION OPTIONS]", divider=st_color)
        with st.form("visu_form"):
            color_map_type = st.selectbox("Color Map",
                                        COLOR_MAP_OPTIONS,
                                        key="color_map")
            alpha_value = st.slider('Mask Transparancy', 0.0, 1.0, 0.5)
            alpha_value = 1 - alpha_value
            if st.toggle('Add Contours to Regions'):
                contour_width = st.slider("Contour width",
                                            min_value=3,
                                            max_value=7,
                                            label_visibility="collapsed")

            else:
                contour_width = None

            show_mask_V = st.selectbox('Display the entropy filter',
                                        options=['PD', 'GT', False],
                                        key="use_mask_V")

            submitted = st.form_submit_button("Submit")

    # Define type of operation
    op_func = np.sum

    if corr_type == 'spearman':
        corr_func = scipy.stats.spearmanr
    elif corr_type == 'pearson':
        corr_func = scipy.stats.pearsonr

    if color_map_type == "viridis":
        color_map = cv2.COLORMAP_VIRIDIS
    elif color_map_type == "cividis":
        color_map = cv2.COLORMAP_CIVIDIS
    elif color_map_type == "magma":
        color_map = cv2.COLORMAP_MAGMA

    # - Loop over images/annotations in sequence ---
    if 'fdx' not in st.session_state:
        st.session_state['fdx'] = 1

    st.header("", divider=st_color)
    st.header("- Contol FloW")
    col_start, _, col_stats = st.columns([1, 1, 4])

    fdx = st.slider('Frame idx (fdx) slider',
                    min_value=0,
                    max_value=len(gt_with_a_counterpart_pd)-1,
                    step=1,
                    value=(st.session_state['fdx'] if st.session_state['fdx']
                           <= len(gt_with_a_counterpart_pd)-1 else
                           len(gt_with_a_counterpart_pd)-1),
                    label_visibility="collapsed",
                    key="fdx_slider",
                    on_change=lambda: st.session_state.update(fdx=st.session_state['fdx_slider']))

    with st.sidebar:
        st.subheader(f":o: :{st_color}[Frame id: {fdx}]")

    with col_start:
        st.subheader(f"Sequence: :{st_color}[{sequence_name}]")
        st.subheader(f"Object id: :{st_color}[{obx}]")
        st.subheader(f"Frame id: :{st_color}[{fdx}]")
        fdx_number = st.number_input('Enter frame index',
                                     min_value=0,
                                     max_value=len(gt_with_a_counterpart_pd)-1,
                                     value=(st.session_state['fdx'] if st.session_state['fdx']
                                            <= len(gt_with_a_counterpart_pd)-1 else len(gt_with_a_counterpart_pd)-1),
                                     label_visibility="collapsed",
                                     key="fdx_number",
                                     on_change=lambda: st.session_state.update(fdx=st.session_state['fdx_number']))

    # Ensure both widgets are synchronized with the session state
    if fdx_number != st.session_state.fdx:
        st.session_state.fdx = fdx_number

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

    # Check if object is present in GT and in PD - frame level
    Obj_in_GT_flag = obx in Nbr_of_objx_in_GT
    Obj_in_PD_flag = obx in Nbr_of_objx_in_PD

    # Extract TP, FN, FP and FN regions per frame per object
    TP, TN, FP, FN = extract_TPTNFPFN(confusions_r, fdx, obx)

    # Color the image with the masks (alpha value)
    im = add_mask_to_im(im, TP, YELLOW, alpha_value)
    im = add_mask_to_im(im, FP, SALMON, alpha_value)
    im = add_mask_to_im(im, FN, BLUE, alpha_value)

    # Color the image with the contours
    gt_for_obj = gt_palette == obx
    if contour_width is not None:
        gt_contours = image_api.get_contours(gt_for_obj, width=contour_width)
        TP_contours = image_api.get_contours(TP, width=contour_width)
        FP_contours = image_api.get_contours(FP, width=contour_width)
        FN_contours = image_api.get_contours(FN, width=contour_width)
        im = add_contours_to_im(im, TP_contours, YELLOW)
        im = add_contours_to_im(im, FP_contours, SALMON)
        im = add_contours_to_im(im, FN_contours, BLUE)
        im = add_contours_to_im(im, gt_contours, GREEN)

    # Adapt efficient entropy for OpenCV - from float64 to uint8
    H_gray = (entropy_frame_fdx*255).astype(np.uint8)
    H_viridis = cv2.applyColorMap(H_gray, color_map)

    # Indicate on the Entropy image, which mask is used when filtering with the mask:
    if show_mask_V:
        # Take X% of the object area
        if show_mask_V == "GT":
            Elem = TP+FN
            object_size = TP.sum() + FN.sum()
        elif show_mask_V == "PD":
            Elem = TP+FP
            object_size = TP.sum() + FP.sum()
        Elem = Elem.astype(np.uint8)

        dilate_flag = st.toggle('Dilate-Erode', value=True)
        if dilate_flag:
            cv2_operation = cv2.dilate
            object_size_X_percent = (5/100)*object_size
        else:
            cv2_operation = cv2.erode
            object_size_X_percent = 10
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
        one_fourth_of_the_array = in_circle.reshape(center_of_kernel[0], -1)
        kernel[center_of_kernel[0]:, center_of_kernel[0]:] = one_fourth_of_the_array
        kernel[center_of_kernel[0]:, :center_of_kernel[0]] = one_fourth_of_the_array[:,::-1]
        kernel[:center_of_kernel[0], :] = kernel[center_of_kernel[0]:, :][::-1]
        # apply the mask on
        dilated_mask = cv2_operation(Elem, kernel, iterations=1).astype(bool)
        mask_V_contours = image_api.get_contours(dilated_mask, width=5)
        masked_H_viridis = H_gray.copy() * dilated_mask
        masked_H_viridis = cv2.applyColorMap(masked_H_viridis, color_map)
        masked_H_viridis = add_contours_to_im(masked_H_viridis,
                                              mask_V_contours,
                                              GREEN)
        H_viridis = add_contours_to_im(H_viridis,
                                       mask_V_contours,
                                       GREEN)
    else:
        masked_H_viridis = ((FP + FN)*255).astype(np.uint8)
        masked_H_viridis = cv2.applyColorMap(masked_H_viridis, color_map)

    # Resize image along the width dimension:
    im = resize(im, NEW_WIDTH)
    H_viridis = resize(H_viridis, NEW_WIDTH)
    masked_H_viridis = resize(masked_H_viridis, NEW_WIDTH)

    st.write("#")
    # st.header("", divider="red")
    st.header("- Visualization")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Image x mask")
        st.image(im, channels="BGR")

    with col2:
        st.subheader("Total entropy")
        st.image(H_viridis, channels="BGR")

    with col3:
        if use_sim:
            st.subheader(f"Errors for object :{st_color}[{obx}]")
        else:
            st.subheader("Masked efficient entropy")
        st.image(masked_H_viridis, channels="BGR")

    # - Plot section ---
    if obx != 0:
        # Load dataframe
        parquet_filename = os.path.join(parquet_file_location,
                                        f"{sequence_name}.parquet")
        try:
            df = pl.read_parquet(parquet_filename)
        except FileNotFoundError:
            st.error(f"File: :orange[{parquet_filename}] not found")
            st.header("No stats for this sequence yet :sweat_smile: ")
            st.stop()

        df_obx = df.filter(df["object_id"] == f"object_{obx}")
        curves = extract_curves(df_obx)

        if use_sim:
            curves = adapt_dataframe_with_simulation(curves)
        if operation_type == "mean":
            curves = adapt_dataframe_with_mean_operation(curves)
        if use_object_size is not False:
            if operation_type == "sum":
                curves = adapt_dataframe_with_object_size(curves,
                                                          use_object_size)
        if use_image_size:
            if operation_type == "sum":
                curves = adapt_dataframe_with_image_size(curves)

        performance_curves = extract_performance_related_curves(curves)
        entropy_curves = extract_entropy_related_curves(curves)
        presence_curves = extract_presence_related_curves(curves)

        # shift the entropy_curves, and presenc_curves
        if shifting_by != 0:
            shifting_value = abs(shifting_by)
            extrapolated_points = np.ones([shifting_value])*np.nan
            for _curves in (entropy_curves, presence_curves):
                for k,v in _curves.items():
                    if shifting_by < 0:
                        new_v = v[shifting_value:]
                        new_v = np.append(new_v, extrapolated_points)
                    elif shifting_value > 0:
                        new_v = v[:-shifting_value]
                        new_v = np.append(extrapolated_points, new_v)
                    _curves[k] = new_v

        with st.sidebar:
            st.write("#")
            st.header(f":{st_color}[PLOTTING]",
                      divider=st_color)

            performance_threshold = st.slider('Failure threshold',
                                              min_value=0.0,
                                              max_value=1.0,
                                              step=0.1,
                                              value=prompt_iou if prompt_history is not None else 0.5,
                                              key="IoU_slider")

            perf_options = list(performance_curves.keys())
            performance_options = st.multiselect("Performance metric",
                                                 options=perf_options,
                                                 default=["IoU"])

            entropie_options = list(entropy_curves.keys())

            col_o, col_x, col_y, col_z = st.columns(4)
            # Initialize session states if they're not already set
            if 'show_H_base' not in st.session_state:
                st.session_state.show_H_base = False
            if 'show_gt_masked' not in st.session_state:
                st.session_state.show_gt_masked = False
            if 'show_pd_masked' not in st.session_state:
                st.session_state.show_pd_masked = False

            # Function to filter options based on a condition
            def filter_options(condition, _options_):
                return [option for option in _options_ if condition in option]

            # Listen for the button clicks and update session state accordingly
            button_show_all_entropy = False
            button_only_base_entropy = False
            button_GT_masked_entropy = False
            button_PD_masked_entropy = False
            with col_o:
                if st.button("Show all", key="show_all_H_base_button"):
                    st.session_state['entropy_options'] = entropie_options
                    button_show_all_entropy = True
            with col_x:
                if st.button("Only Base", key="show_H_base_button"):
                    st.session_state['entropy_options'] = filter_options("base", entropie_options)
                    button_only_base_entropy = True
            with col_y:
                if st.button("GT masked", key="show_gt_masked_button"):
                    st.session_state['entropy_options'] = filter_options("masked_gt", entropie_options)
                    button_GT_masked_entropy = True
            with col_z:
                if st.button("PD masked", key="show_pd_masked_button"):
                    st.session_state['entropy_options'] = filter_options("masked_pd", entropie_options)
                    button_PD_masked_entropy = True

            default_values = st.session_state.get('entropy_options', [])

            reinit_button_colors = False
            if default_values == []:
                reinit_button_colors = True

            if use_object_size == False:
                default_values = ["Total_H_base"] if default_values == [] else default_values
            elif "GT" in use_object_size:
                default_values = ["Total_H_masked_gt"] if default_values == [] else default_values
            elif "PD" in use_object_size:
                default_values = ["Total_H_masked_pd"] if default_values == [] else default_values

            # Compute the curves dissaperances also based on the threshold
            Failure_regions = performance_curves["IoU"] > performance_threshold
            presence_curves.update({"Failure": Failure_regions})

            with st.form("entropy_plot_form"):
                entropy_options = st.multiselect("Entropy",
                                                options=entropie_options,
                                                default=default_values,
                                                key="entropy_options")
                _presence_options = presence_curves.keys()
                presence_options = st.multiselect("Show disaperances",
                                                options=_presence_options,
                                                default=_presence_options)

                submitted = st.form_submit_button("Submit")

            # Update the session state based on the multiselect interaction
            # This ensures any manual update to the selection clears the specific defaults
            if entropy_options != default_values:
                st.session_state.show_H_base = False
                st.session_state.show_gt_masked = False
                st.session_state.show_pd_masked = False

            if button_show_all_entropy:
                ChangeButtonColour("Show all")
                RevertButtonColour("Only Base")
                RevertButtonColour("GT masked")
                RevertButtonColour("PD masked")
            elif button_only_base_entropy:
                RevertButtonColour("Show all")
                ChangeButtonColour("Only Base")
                RevertButtonColour("GT masked")
                RevertButtonColour("PD masked")
            elif button_GT_masked_entropy:
                RevertButtonColour("Show all")
                ChangeButtonColour("GT masked")
                RevertButtonColour("Only Base")
                RevertButtonColour("PD masked")
            elif button_PD_masked_entropy:
                RevertButtonColour("Show all")
                ChangeButtonColour("PD masked")
                RevertButtonColour("GT masked")
                RevertButtonColour("Only Base")

            if reinit_button_colors:
                RevertButtonColour("Show all")
                RevertButtonColour("PD masked")
                RevertButtonColour("GT masked")
                RevertButtonColour("Only Base")

        # Create an empty figure with a title
        fig = go.Figure(layout=go.Layout(title=sequence_name,
                                         font=dict(size=18)))
        frame_indices = list(range(len(df_obx["frame"])))
        # Disapperances curve
        for curve_key in presence_options:
            draw_span_plotly(fig,
                             frame_indices,
                             presence_curves[curve_key],
                             color_dict[curve_key])

        # Add performance curve
        for curve_key in performance_options:
            fig.add_trace(go.Scatter(x=frame_indices,
                                     y=performance_curves[curve_key],
                                     mode='lines',
                                     name=curve_key,
                                     marker=dict(color=color_dict[curve_key])
                                     ))
            # Robustness 
            #fig.add_hline(y=performance_threshold,
            #              line_width=3,
            #              # line_dash="dash",
            #              line_color="green")

        # Configure the layout to include a secondary y-axis on the right
        fig.update_layout(
            yaxis=dict(
                title=r"IoU",
                range=[0.0, 1.0]
            ),
            yaxis2=dict(
                title=r"Entropy",
                overlaying='y',
                side='right',
                range=[0.0, 1.0]
            )
        )

        # Add entropy on the secondary y-axis
        for curve_key in entropy_options:
            line_type = {'base': 'solid',
                         'masked_gt': 'dash',
                         'masked_pd': 'solid'}
            line_style = line_type[curve_key.split("_H_")[-1]]
            fig.add_trace(go.Scatter(x=frame_indices,
                                     y=entropy_curves[curve_key],
                                     mode='lines',
                                     name=curve_key,
                                     yaxis='y2',
                                     marker=dict(color=color_dict[curve_key]),
                                     line=dict(dash=line_style)
                                     ))

        # Place the legent on the right-up corner of the graph
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))

        # Slider that follows the frame idx -- fdx.
        # fig.add_vline(x=fdx, line_width=5, line_dash="dash", line_color="red")
        
        st.write("#")
        st.header(f"- Plotting")
        # Add the prompts if available # TODO: CLEAN THESE CODE FOR THE PROMPTS
        if prompt_history is not None:
            interactions_fdx_location = [f for (p, f) in zip(prompt_calls, fdx_prompts) if p]
            for p_fdx in interactions_fdx_location:
                fig.add_vline(x=p_fdx, line_width=3, line_dash="dash",
                              line_color="#01c5c4")
            fig.add_trace(go.Scatter(x=fdx_prompts,
                                     y=entropy_prompts,
                                     mode='lines',
                                     name=curve_key+" before prompt",
                                     yaxis='y2',
                                     marker=dict(color="#ffffff", size=10),
                                     line=dict(dash=line_style)
                                     ))
            if iou_prompts is not None:
                # The IoU recoded (shown if available)
                fig.add_trace(go.Scatter(x=fdx_prompts,
                                        y=iou_prompts,
                                        mode='lines',
                                        name="recorded IoU before prompt",
                                        yaxis='y2',
                                        marker=dict(color="#4bff4b", size=10),
                                        line=dict(dash=line_style)))

            if "Threshold" in list(meta_prompts_df.columns):
                if not prompt_ent == 1000.0:
                    fig.add_hline(y=prompt_ent, line_width=3, yref='y2',
                                  line_dash="longdash", line_color="#FFF000")
                    st.subheader(f":guardsman: :red[Entropy Threshold] to issue a prompt :red[{prompt_ent}]")

        if (entropy_options != []) or (performance_options != []):
            st.subheader(f"Line plot: :{st_color}[Performance] and :{st_color}[Entropy] variations wrt to sequence length")
            if use_object_size != False: 
                st.subheader(f":warning: Decoupling from {use_object_size} object size")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.header(":construction: :orange[No curves to show] :construction:")

        # ---------------------------------------------------------------------
        # - Take a look at the derivatives ---
        derivative_entropy = entropy_curves.copy()
        jerk_entropy = entropy_curves.copy()
        derivative_perf = performance_curves.copy()

        # Compute all derivatives
        y_min_der=0.0
        y_max_der=0.0
        for curve_key in entropy_curves.keys():
            entropy_curve = entropy_curves[curve_key]
            derivative = derive(entropy_curve)
            derivative_entropy[curve_key] = derivative
            jerk = derive(derivative)
            jerk_entropy[curve_key] = jerk
            y_min_der = min(y_min_der, derivative.min()) # only for visualization purposes
            y_max_der = max(y_max_der, derivative.max()) # only for visualization purposes
            

        for perf_key in performance_curves.keys():
            perf_curve = performance_curves[perf_key]
            derivative = derive(perf_curve)
            derivative_perf[perf_key] = derivative
            y_min_der = min(y_min_der, derivative.min()) # only for visualization purposes
            y_max_der = max(y_max_der, derivative.max()) # only for visualization purposes

        if use_derivatives:
            fig = go.Figure(layout=go.Layout(title=sequence_name,
                                             font=dict(size=18)))
            frame_indices = list(range(len(df_obx["frame"])))
            # Disapperances curve
            for curve_key in presence_options:
                draw_span_plotly(fig,
                                 frame_indices,
                                 presence_curves[curve_key],
                                 color_dict[curve_key],
                                 y_min = y_min_der,
                                 y_max = y_max_der)

            # Add performance curve
            for curve_key in performance_options:
                fig.add_trace(go.Scatter(x=frame_indices,
                                         y=derivative_perf[curve_key],
                                         mode='lines',
                                         name=curve_key,
                                         marker=dict(color=color_dict[curve_key])
                                         ))

            # Configure the layout to include a secondary y-axis on the right
            fig.update_layout(yaxis=dict(title="IoU and Entropy derivates"))

            # Add derivate on the secondary y-axis
            for curve_key in entropy_options:
                line_type = {'base': 'solid',
                             'masked_gt': 'dash',
                             'masked_pd': 'dot'}
                line_style = line_type[curve_key.split("_H_")[-1]]
                derivative = derivative_entropy[curve_key]
                fig.add_trace(go.Scatter(x=frame_indices,
                                         y=derivative,
                                         mode='lines',
                                         name=curve_key,
                                         marker=dict(color=color_dict[curve_key]),
                                         line=dict(dash=line_style)
                                         ))
                fig.add_hline(y=0,
                line_width=3,
                line_dash="dash",
                line_color="red")

            # Place the legent on the right-up corner of the graph
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))

            # Slider that follows the frame idx -- fdx.
            fig.add_vline(x=fdx, line_width=5, line_dash="dash",
                          line_color="red")

            st.write("#")
            st.subheader(f"- :{st_color}[Derivates]")
            if (entropy_options != []) or (performance_options != []):
                if use_object_size != False: 
                    st.subheader(f":warning: Decoupling from {use_object_size} object size")
                st.plotly_chart(fig, theme=None, use_container_width=True)
            else:
                st.header(":construction: :orange[No curves to show] :construction:")
                
                
            # Use the Jerk
            fig = go.Figure(layout=go.Layout(title=sequence_name,
                                             font=dict(size=18)))
            
            for curve_key in entropy_options:
                line_type = {'base': 'solid',
                             'masked_gt': 'dash',
                             'masked_pd': 'dot'}
                line_style = line_type[curve_key.split("_H_")[-1]]
                jerk = jerk_entropy[curve_key]
                fig.add_trace(go.Scatter(x=frame_indices,
                                         y=jerk,
                                         mode='lines',
                                         name=curve_key,
                                         marker=dict(color=color_dict[curve_key]),
                                         line=dict(dash=line_style)
                                         ))
                fig.add_hline(y=0,
                line_width=3,
                line_dash="dash",
                line_color="red")

            # Place the legent on the right-up corner of the graph
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            
            st.write("#")
            st.subheader(f"- :{st_color}[Jerk]")
            if (entropy_options != []) or (performance_options != []):
                if use_object_size != False: 
                    st.subheader(f":warning: Decoupling from {use_object_size} object size")
                st.plotly_chart(fig, theme=None, use_container_width=True)
            else:
                st.header(":construction: :orange[No curves to show] :construction:")

        # -------------------------------------------------------------

        # See size variations
        size_curves = extract_size_related_curves(curves)
        with st.sidebar:
            _options = list(size_curves.keys())[1:]  # Discarding the "Image_size" key

            # Initialize session state for 'show_all_clicked' if it's not already set
            if 'show_all_clicked' not in st.session_state:
                st.session_state.show_all_clicked = False

            # Listen for the "Show all" button click and update session state
            if st.button("Show sizes", key="show_all_sized"):
                st.session_state.show_all_clicked = True
                st.session_state['size_options'] = _options  # Pre-select all options

            # Determine default values based on the session state
            default_values = st.session_state.get('size_options', [])

            # Multiselect for size options
            with st.form("size_plot_form"):
                size_options = st.multiselect("Show size variations",
                                            options=_options,
                                            default=default_values,
                                            key="size_options")

                submitted = st.form_submit_button("Submit")

            # If the user modifies the multiselect, update the session state to reflect this change
            if size_options != default_values:
                st.session_state.show_all_clicked = False  # Reset the flag if selections are modified
                st.session_state['size_options'] = size_options  # Update the session state with the new selection

        if size_options:
            fig_bis = go.Figure(layout=go.Layout(title="Size variations",
                                                 font=dict(size=18)))
            for curve_key in presence_options:
                draw_span_plotly(fig_bis,
                                 frame_indices,
                                 presence_curves[curve_key],
                                 color_dict[curve_key])

            for curve_key in size_options:
                fig_bis.add_trace(go.Scatter(x=frame_indices,
                                             y=size_curves[curve_key]/size_curves["Image_size"],
                                             mode='lines',
                                             name=curve_key,
                                             marker=dict(color=color_dict[curve_key])
                                             ))

            # Place the legent on the right-up corner of the graph
            fig_bis.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))

            # Slider that follows the frame idx -- fdx.
            fig_bis.add_vline(x=fdx, line_width=5, line_dash="dash",
                              line_color="red")

            st.plotly_chart(fig_bis, theme=None, use_container_width=True)

        st.header("- Summary Statistics")

        corr_coeffs = {}
        p_values = {}
        # TODO: Extract this method
        # Compute the correlation coefficients for entropy curve
        for curve_key in entropy_curves.keys():
            if corr_type == 'spearman':
                # Warning ! Discarding the fist frame, because not a prediction
                # Similarly to DAVIS 2017
                S_correlation, p_value = corr_func(df_obx["IoU"][1:],
                                                   entropy_curves[curve_key][1:],
                                                   nan_policy='omit')
            elif corr_type == 'pearson':
                S_correlation, p_value = corr_func(df_obx["IoU"][1:],
                                                   entropy_curves[curve_key][1:])

            corr_coeffs.update({curve_key: S_correlation})
            p_values.update({curve_key: p_value})

        # Separate the correlation coefficients
        base_corr_coeffs = {}
        base_p_values = {}
        for k, v in corr_coeffs.items():
            if ('_base') not in k:
                continue
            kk = k.split('_base')[0]
            base_corr_coeffs.update({kk: np.round(v, 3)})
            base_p_values.update({kk: '{:.2g}'.format(p_values[k])})

        masked_gt_corr_coeffs = {}
        masked_gt_p_values = {}
        for k, v in corr_coeffs.items():
            if ('_masked_gt') not in k:
                continue
            kk = k.split('_masked_gt')[0]
            masked_gt_corr_coeffs.update({kk: np.round(v, 3)})
            masked_gt_p_values.update({kk: '{:.2g}'.format(p_values[k])})

        masked_pd_corr_coeffs = {}
        masked_pd_p_values = {}
        for k, v in corr_coeffs.items():
            if ('_masked_pd') not in k:
                continue
            kk = k.split('_masked_pd')[0]
            masked_pd_corr_coeffs.update({kk: np.round(v, 3)})
            masked_pd_p_values.update({kk: '{:.2g}'.format(p_values[k])})

        custom_order = ["Total_H",
                        "TP_H", "TN_H", "FP_H", "FN_H", "TR_H", "FR_H"]
        if use_sim:
            total_corr_coeffs = {"Base": base_corr_coeffs}
            total_p_values = {"Base": base_p_values}
        else:
            total_corr_coeffs = {"Base": base_corr_coeffs,
                                 "Masked GT": masked_gt_corr_coeffs,
                                 "Masked PD": masked_pd_corr_coeffs}
            total_p_values = {"Base": base_p_values,
                              "Masked GT": masked_gt_p_values,
                              "Masked PD": masked_pd_p_values}

        # ----------------------------------
        if use_derivatives:
            deriv_corr_coeffs = {}
            deriv_p_values = {}
            # TODO: Extract this method
            # Compute the correlation coefficients for the derivate curve
            for deriv_curve_key in derivative_entropy.keys():
                if corr_type == 'spearman':
                    # Warning ! Discarding the fist frame, because not a prediction
                    # Similarly to DAVIS 2017
                    deriv_S_correlation, deriv_p_value = corr_func(df_obx["IoU"][1:],
                                                                   derivative_entropy[deriv_curve_key][1:],
                                                                   nan_policy='omit')
                elif corr_type == 'pearson':
                    deriv_S_correlation, deriv_p_value = corr_func(df_obx["IoU"][1:],
                                                                   derivative_entropy[deriv_curve_key][1:])

                deriv_corr_coeffs.update({deriv_curve_key: deriv_S_correlation})
                deriv_p_values.update({deriv_curve_key: deriv_p_value})

            # Separate the correlation coefficients
            deriv_base_corr_coeffs = {}
            deriv_base_p_values = {}
            for k, v in deriv_corr_coeffs.items():
                if ('_base') not in k:
                    continue
                kk = k.split('_base')[0]
                deriv_base_corr_coeffs.update({kk: np.round(v, 3)})
                deriv_base_p_values.update({kk: '{:.2g}'.format(p_values[k])})

            deriv_masked_gt_corr_coeffs = {}
            deriv_masked_gt_p_values = {}
            for k, v in deriv_corr_coeffs.items():
                if ('_masked_gt') not in k:
                    continue
                kk = k.split('_masked_gt')[0]
                deriv_masked_gt_corr_coeffs.update({kk: np.round(v, 3)})
                deriv_masked_gt_p_values.update({kk: '{:.2g}'.format(p_values[k])})

            deriv_masked_pd_corr_coeffs = {}
            deriv_masked_pd_p_values = {}
            for k, v in deriv_corr_coeffs.items():
                if ('_masked_pd') not in k:
                    continue
                kk = k.split('_masked_pd')[0]
                deriv_masked_pd_corr_coeffs.update({kk: np.round(v, 3)})
                deriv_masked_pd_p_values.update({kk: '{:.2g}'.format(p_values[k])})

            custom_order = ["Total_H",
                            "TP_H", "TN_H", "FP_H", "FN_H", "TR_H", "FR_H"]
            if use_sim:
                deriv_total_corr_coeffs = {"Base": deriv_base_corr_coeffs}
                deriv_total_p_values = {"Base": deriv_base_p_values}
            else:
                deriv_total_corr_coeffs = {"Base": deriv_base_corr_coeffs,
                                        "Masked GT": deriv_masked_gt_corr_coeffs,
                                        "Masked PD": deriv_masked_pd_corr_coeffs}
                deriv_total_p_values = {"Base": deriv_base_p_values,
                                        "Masked GT": deriv_masked_gt_p_values,
                                        "Masked PD": deriv_masked_pd_p_values}
            # ----------------------------------

        col1, col2 = st.columns(2)
        with col_stats:
            # Entropy values
            st.subheader(f":{st_color}[{corr_type}] - Correlation Coefficients")
            df_corr = pd.DataFrame(total_corr_coeffs).T
            df_corr = df_corr[custom_order]  # Imposse my order
            st.dataframe(df_corr, width=2000)

            st.subheader(" and p values")
            df_p = pd.DataFrame(total_p_values).T
            df_p = df_p[custom_order]  # Imposse my order
            st.dataframe(df_p, width=2000)

            if use_derivatives:
                # Derivatives of the entropy curves
                st.subheader(f":{st_color}[{corr_type}] - Derivative Correlation Coefficients")
                deriv_df_corr = pd.DataFrame(deriv_total_corr_coeffs).T
                deriv_df_corr = deriv_df_corr[custom_order]  # Imposse my order
                st.dataframe(deriv_df_corr, width=2000)

                st.subheader(" and p values")
                deriv_df_p = pd.DataFrame(deriv_total_p_values).T
                deriv_df_p = deriv_df_p[custom_order]  # Imposse my order
                st.dataframe(deriv_df_p, width=2000)

        with col1:
            st.subheader(f":{st_color}[{corr_type}] - Correlation Coefficients and p values")

            fig, (ax, ax_p) = plt.subplots(2, 1, figsize=(5, 1 if use_sim else 2))  # Adjust the size as needed

            # Hide axes
            ax.axis('off')
            ax_p.axis('off')

            # Table from DataFrame
            table = ax.table(cellText=df_corr.values, colLabels=df_corr.columns,
                             rowLabels=df_corr.index, loc='center', cellLoc='center')

            # Make the text bold
            for (i, j), cell in table.get_celld().items():
                cell.set_text_props(fontweight='bold')

            # Auto-set the column widths to make the content fit well
            table.auto_set_column_width(col=list(range(len(df_corr.columns))))

            # Table from DataFrame
            table_p = ax_p.table(cellText=df_p.values, colLabels=df_p.columns,
                                 rowLabels=df_p.index, loc='center', cellLoc='center')

            # Make the text bold
            for (i, j), cell in table_p.get_celld().items():
                cell.set_text_props(fontweight='bold')

            # Auto-set the column widths to make the content fit well
            table_p.auto_set_column_width(col=list(range(len(df_p.columns))))

            # Adjust layout to make room for table
            fig.tight_layout()

            # Show the table in Streamlit
            st.pyplot(fig)

            if use_derivatives:
                st.subheader(f":{st_color}[Derivative's {corr_type}] - Correlation Coefficients and p values")

                fig, (ax, ax_p) = plt.subplots(2, 1, figsize=(5, 1 if use_sim else 2))  # Adjust the size as needed

                # Hide axes
                ax.axis('off')
                ax_p.axis('off')

                # Table from DataFrame
                table = ax.table(cellText=deriv_df_corr.values, colLabels=deriv_df_corr.columns,
                                 rowLabels=deriv_df_corr.index, loc='center', cellLoc='center')

                # Make the text bold
                for (i, j), cell in table.get_celld().items():
                    cell.set_text_props(fontweight='bold')

                # Auto-set the column widths to make the content fit well
                table.auto_set_column_width(col=list(range(len(deriv_df_corr.columns))))

                # Table from DataFrame
                table_p = ax_p.table(cellText=deriv_df_p.values, colLabels=deriv_df_p.columns,
                                     rowLabels=deriv_df_p.index, loc='center', cellLoc='center')

                # Make the text bold
                for (i, j), cell in table_p.get_celld().items():
                    cell.set_text_props(fontweight='bold')

                # Auto-set the column widths to make the content fit well
                table_p.auto_set_column_width(col=list(range(len(deriv_df_p.columns))))

                # Adjust layout to make room for table
                fig.tight_layout()

                # Show the table in Streamlit
                st.pyplot(fig)

        with col2:
            fig = go.Figure(layout=go.Layout(title=sequence_name,
                                             font=dict(size=18)))

            # Add entropy on the secondary y-axis
            for curve_key in entropy_options:
                markser_style = {'base': 'circle',
                                 'masked_gt': 'star',
                                 'masked_pd': 'diamond'}
                markser_style = markser_style[curve_key.split("_H_")[-1]]
                fig.add_trace(go.Scatter(x=df_obx["IoU"],
                                         y=entropy_curves[curve_key],
                                         mode='markers',
                                         name=curve_key,
                                         marker=dict(
                                             color=color_dict[curve_key],
                                             size=13,
                                             symbol=markser_style,
                                             opacity=0.5)
                                         )
                              )

            fig.update_layout(legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="right",
                                          x=1))

            if entropy_options != []:
                st.subheader("Scatter plot: :{default_layout_color}[Entropy] wrt :{default_layout_color}[IoU]")
                st.plotly_chart(fig, theme=None, use_container_width=True)
            else:
                st.subheader(":construction: :orange[No curves to show] :construction: - select entropy curves")

        st.write("#")
        st.header("- Table")
        st.subheader(f"Dataframe of sequence: :{st_color}[{sequence_name}]")
        st.dataframe(df_obx)
