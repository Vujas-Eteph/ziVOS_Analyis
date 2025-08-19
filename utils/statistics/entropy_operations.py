'''
Compute Entropy related stuff

Stéphane Vujasinovic
'''
from typing import Callable, Tuple

import numpy as np
from icecream import ic


class EntropyHelper:
    """
    Handle entropy calculation.
    """
    def __init__(self):
        pass

    @staticmethod
    def __check_if_pmf(
        input: np.ndarray,
        axis: int
    ) -> bool:
        return (1, 1) == (np.round(np.sum(input, axis=axis).min(), 5),
                          np.round(np.sum(input, axis=axis).max(), 5))

    @staticmethod
    def __self_information(
        x: np.ndarray
    ) -> np.ndarray:
        return -1*np.log(x)

    @staticmethod
    def __compute_entropy(
        x: np.ndarray
    ) -> np.ndarray:
        z = np.zeros([1, *x.shape[1:]], dtype=x.dtype)
        for y in x:
            z += y * EntropyHelper.__self_information(y)
        return z

    @staticmethod
    def _squeeze_list_of_arrays(
        input_list: list
    ) -> list:
        return [np.squeeze(element) for element in input_list]

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @mean.setter
    def mean(
        self,
        predictions_list: list
    ):
        assert isinstance(predictions_list, list)
        predictions_array = np.array(EntropyHelper._squeeze_list_of_arrays(
            predictions_list))
        self._mean = predictions_array.mean(axis=0)
        if len(self._mean.shape) < 3:
            self._mean = np.expand_dims(self._mean, 0)

    @property
    def std(self) -> np.ndarray:
        return self._std

    @std.setter
    def std(
        self,
        predictions_list: list
    ):
        assert isinstance(predictions_list, list)
        predictions_array = np.array(EntropyHelper._squeeze_list_of_arrays(
            predictions_list))
        self._std = predictions_array.std(axis=0)
        if len(self._std.shape) < 3:
            self._std = np.expand_dims(self._std, 0)

    @property
    def entropy(self) -> np.ndarray:
        return self._entropy

    @entropy.setter
    def entropy(
        self,
        prediction: np.ndarray
    ):
        assert isinstance(prediction, np.ndarray)
        assert EntropyHelper.__check_if_pmf(prediction, 0)
        self._entropy = EntropyHelper.__compute_entropy(prediction)

    @property
    def norm_entropy(self) -> np.ndarray:
        return self._norm_entropy

    # Compute efficient entropy
    @norm_entropy.setter
    def norm_entropy(
        self,
        prediction: np.ndarray
    ):
        number_of_categories = prediction.shape[0]
        res = self._entropy.copy() / np.log(number_of_categories)
        self._norm_entropy = np.round(res, 4)


def operation_on_Entropy_over_TPTNFPFN(
    h: np.ndarray,
    TP: np.ndarray,
    TN: np.ndarray,
    FP: np.ndarray,
    FN: np.ndarray,
    op_func: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    H.shape: [#frames, 1, H, W]. Hence the second axis can be discarded
    """
    # Trick to avoid meaning with 0. Take the lowest point of the H, as when no filterning nothing happens, but if a filter was used beforehand is now applied...
    # ... Allows to not take into account the 0 during the mean operaition (no influence on the summation).
    # The trick is to filter the values based on the lowet valuea nd give the filtered version to the op_func
    h_filtered = h[~np.isclose(h, 0.0, rtol=1e-09, atol=1e-09)]
    h_min = (np.array([0]) if h_filtered.tolist() == [] else h_filtered).min()

    TP_H = op_func(h[h*TP >= h_min])
    TN_H = op_func(h[h*TN >= h_min])
    FP_H = op_func(h[h*FP >= h_min])
    FN_H = op_func(h[h*FN >= h_min])

    return TP_H, TN_H, FP_H, FN_H


def sum_Entropy_over_TPTNFPFN(
    H: np.ndarray,
    fdx: int,
    TP: np.ndarray,
    TN: np.ndarray,
    FP: np.ndarray,
    FN: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    H.shape: [#frames, 1, H, W]. Hece the second axis can be discarded
    """
    TP_H = (H[fdx, 0, :, :]*TP).sum()
    TN_H = (H[fdx, 0, :, :]*TN).sum()
    FP_H = (H[fdx, 0, :, :]*FP).sum()
    FN_H = (H[fdx, 0, :, :]*FN).sum()

    return TP_H, TN_H, FP_H, FN_H
