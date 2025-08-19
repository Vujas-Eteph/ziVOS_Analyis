'''
Collection of functions to help to compute metrics

by Stéphane Vujasinovic
'''
from collections import Counter

import numpy as np


def compute_IoU(
    tp: np.ndarray,
    tn: np.ndarray,  # not used, but kept for consistency
    fp: np.ndarray,
    fn: np.ndarray
) -> float:
    if tp.sum == 0:
        return 0.0
    return tp.sum()/(tp.sum()+fp.sum()+fn.sum())  # IoU


def compute_VOTS_metrics(IoU_obx_curve, vots_successes, vots_cases):
    # TODO: Check that also for the successes and VOTS cases we discard the first frame and last frame
    vots_tracking_success_ratio, vots_tracking_quality = \
        compute_VOTS_metrics_primary_metrics(vots_successes)
    vots_accuracy, vots_robustness, vots_nre, vots_dre, vots_adq = \
        compute_VOTS_metrics_secondary_metrics(IoU_obx_curve, vots_successes, vots_cases)

    return vots_tracking_success_ratio, vots_tracking_quality, vots_accuracy, \
        vots_robustness, vots_nre, vots_dre, vots_adq


def compute_VOTS_metrics_primary_metrics(vots_successes):
    vots_tracking_success_ratio = \
        vots_successes.sum() / len(vots_successes)
    vots_tracking_quality = None
    # TODO : Compute Q -> Is the AUR see document JOR
    return vots_tracking_success_ratio, vots_tracking_quality


def compute_VOTS_metrics_secondary_metrics(IoU_obx_curve, vots_successes, vots_cases):
    count_occurrences_1 = Counter(np.array(vots_cases)[np.isclose(vots_successes,
                                                        1.0,
                                                        rtol=1e-09,
                                                        atol=1e-09)])
    vots_accuracy = IoU_obx_curve[np.array(vots_cases)[np.isclose(vots_successes,
                                                        1.0,
                                                        rtol=1e-09,
                                                        atol=1e-09)] == 'A'].mean()
    vots_robustness = count_occurrences_1['A']
    count_occurrences_2 = Counter(vots_cases)
    vots_nre = count_occurrences_2['C'] / len(vots_successes)
    vots_dre = count_occurrences_2['B'] / len(vots_successes)
    vots_adq = count_occurrences_2['E'] / len(vots_successes)

    return vots_accuracy, vots_robustness, vots_nre, vots_dre, vots_adq


def extract_metrics(tp, tn, fp, fn, obj_in_gt_flag, obj_in_pd_flag,
                    io_u_lower_bound=0):
    # Definitions
    vots_success = None
    vots_case = None
    if obj_in_gt_flag and obj_in_pd_flag:  # VOTS case A and B
        iou = compute_IoU(tp, tn, fp, fn)
        if iou > io_u_lower_bound:  # VOTS success
            vots_success = 1
        if iou > 0:  # VOTS case A
            vots_case = 'A'
        if iou == 0:  # VOTS case B
            vots_success = 0
            vots_case = 'B'
    elif obj_in_gt_flag and not obj_in_pd_flag:  # VOTS case C
        # Object is missing from the prediction
        iou = 0.0
        vots_success = 0
        vots_case = 'C'
    elif not obj_in_gt_flag and obj_in_pd_flag:  # VOTS case D
        # Object is missing from the GT, but method still predicted
        # an object...
        iou = 0.0
        vots_success = 0
        vots_case = 'D'
    elif not obj_in_gt_flag and not obj_in_pd_flag:  # VOTS case E
        # The method correlty predicted a missing object
        iou = 1.0  # In compliance with VOT metrics
        vots_success = 1
        vots_case = 'E'
    else:
        raise ('Case not covered...')

    return iou, vots_success, vots_case


def compute_Dice(
    TP: np.ndarray,
    TN: np.ndarray,
    FP: np.ndarray,
    FN: np.ndarray
) -> float:
    """Compute the Sørensen–Dice coefficient"""
    Dice_score = (2*TP.sum())/(2*TP.sum() + FP.sum() + FN.sum())

    return Dice_score
