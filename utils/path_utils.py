"""
Functions for common stuff.

Author: Stephane Vujasinovic
"""

# - IMPORTS ---
import os


# - FUNCTIONS ---
def create_directory_if_not_there(
    path: str
):
    '''
    Create a directory and all intermediate directories if they don't exist.
    '''
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        raise Exception(f"An error occurred while creating the directory: {e}") 


def check_existence_of_directory(
    path: str
):
    '''
    Check if the directory exists
    '''
    return os.path.exists(path) and os.path.isdir(path)


def find_subdirectories(
    directory: str
):
    '''
    Find the subdirectories
    '''
    try:
        return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
    except OSError as e:
        raise Exception(f"An error occurred while accessing the directory: {e}")


# - FILTERS FUNCTIONS ---
def get_files_from_sequence(
    directory_to_look_into: str,
    sequence: str
) -> str:
    return os.listdir(os.path.join(directory_to_look_into, sequence))


def filter_and_sort_filenames(
    gt_masks_file_names: list,
    file_names: list,
    extension_to_replace: str,
    target_extension='.png'
) -> list:
    return sorted([filename for filename in file_names if filename.replace(extension_to_replace, target_extension) in gt_masks_file_names])


def extract_common_elems(
    list1: list,
    list2: list
) -> list:
    return [item for item in list1 if item in list2]


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
