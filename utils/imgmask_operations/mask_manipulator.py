'''
API for mask operations

by Stephane Vujasinovic
'''
import warnings
from typing import Tuple

import numpy as np
import scipy
import skimage
from icecream import ic


class MaskManipulator():
    def __init__(self):
        pass


def extract_contours(
    mask: np.ndarray,
    width=5
)-> np.ndarray:
    """Get mask of contours. Does not make sense to be as we discard the Epistemic and Aleatoric uncertainties.

    Args:
        mask (_type_): _description_
    """
    og_shape = mask.shape
    batch_length = np.prod(og_shape[:3])
    new_mask = np.reshape(mask, (batch_length, *og_shape[3:]))

    structuring_element = np.array([
                                    [0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]
                                ], dtype=bool)

    structuring_element = np.ones([width, width], dtype=bool)

    dilated_mask = np.array([scipy.ndimage.binary_dilation(slice, structuring_element) for slice in new_mask])
    eroded_mask = np.array([scipy.ndimage.binary_erosion(slice, structuring_element) for slice in new_mask])
    stack_of_contours = dilated_mask ^ eroded_mask  # XOR operation, as - not suported for boolean np.arrays

    return np.reshape(stack_of_contours, og_shape)


def apply_filtering_technique(
    entropy_arr: np.ndarray,
    filtering_technique=None,
    **kwargs
) -> np.array:
    if filtering_technique == 'mean':
        limit_x = entropy_arr.mean()
    elif filtering_technique == 'std':
        limit_x = entropy_arr.std()
    elif filtering_technique == 'lower_bound':
        limit_x = kwargs.get('lower_bound', 0.2)
    elif filtering_technique == 'ransac':
        print('\nnot implemented yet')
    elif filtering_technique == 'mean_TN':
        TN = kwargs.get('TN', 1)
        limit_x = (entropy_arr*TN).mean()
    elif filtering_technique == 'std_TN':
        TN = kwargs.get('TN', 1)
        if TN == 1:
            warnings.warn(f"filering_technique_flag is {filtering_technique}",
                            "but True Negative is not given")
        limit_x = (entropy_arr*TN).std()
    else:
        raise('Error filtering_technique not supported')

    ic(filtering_technique, limit_x)
    entropy_arr[entropy_arr <= limit_x] = 0.0

    return entropy_arr


def filter_entropy(entropy_r_fdx, filtering_technique, **kwargs):
    # Apply filtering technique
    return apply_filtering_technique(entropy_r_fdx,
                                     filtering_technique,
                                     **kwargs)


def adjust_entropy_with_density(entropy_array, tp_h, tn_h, fp_h, fn_h):
    area = np.prod(entropy_array.shape[2:])
    tp_h /= area
    tn_h /= area
    fp_h /= area
    fn_h /= area

    # if args.relative_density:  # TODO: fix - does not work properly
    #     tp_h /= (TP.sum() if TP.sum() != 0 else 1)
    #     tn_h /= (TN.sum() if TN.sum() != 0 else 1)
    #     fp_h /= (FP.sum() if FP.sum() != 0 else 1)
    #     fn_h /= (FN.sum() if FN.sum() != 0 else 1)
    #     area = np.prod(entropy_array.shape[2:])

    return tp_h, tn_h, fp_h, fn_h, area


def extract_TPTNFPFN(
    conf_r: np.ndarray,
    fdx: int,
    obx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    confusions_masks = conf_r[:, fdx, obx, :, :]
    TP = confusions_masks[0]
    TN = confusions_masks[1]
    FP = confusions_masks[2]
    FN = confusions_masks[3]

    return TP, TN, FP, FN


def compute_confusion_array(
    gt_masks: np.ndarray,
    pd_masks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute True Positive (TP), True Negative (TN),
    False Positive (FP) and False Negative (FN) arrays.

    Args:
        gt_stack (np.ndarray): Ground truth binary mask stack.
        mask_stack (np.ndarray): Predicted binary mask stack.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A tuple containing arrays for TP, TN, FP, and FN.
    """
    # Compute confusions
    TP = pd_masks*gt_masks
    TN = (1-pd_masks)*(1-gt_masks)
    FP = pd_masks*(1-gt_masks)
    FN = (1-pd_masks)*(gt_masks)

    return TP, TN, FP, FN


def adjust_TrueNegative_mask_for_missing_obx(
    ids_not_present: list,
    TN: np.ndarray
) -> np.ndarray:
    '''
    If object not present, every corresponding mask is set to false,
    expect the FP, kept intact
    '''
    if [] == ids_not_present:
        return TN

    for obj_idx in ids_not_present:
        TN[obj_idx, :, :] = 0*TN[obj_idx, :, :]

    return TN


def include_the_background(
    exclude_background_from_visual_flag: bool
) -> int:
    if exclude_background_from_visual_flag:
        return 0
    return 1


def get_number_of_objects(
    hkl_score: np.array
) -> int:
    """
    Extract the number of objects present in the sequence (Current frame).
    The background is included.
    """
    return hkl_score.shape[0]
