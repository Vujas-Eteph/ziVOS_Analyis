'''
Collection of functions to help to generate/extract/... confusions arrays.

by Stéphane Vujasinovic
'''
from typing import Tuple

import numpy as np


def extract_TPTNFPFN(
    conf_r: np.ndarray,
    frame_index: int,
    object_index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    confusions_masks = conf_r[:, frame_index, object_index, :, :]
    tp = confusions_masks[0]
    tn = confusions_masks[1]
    fp = confusions_masks[2]
    fn = confusions_masks[3]

    return tp, tn, fp, fn
