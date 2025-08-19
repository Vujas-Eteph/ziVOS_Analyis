"""
Functions for manipulating dataframes

by Stephane Vujasinovic
"""

# - IMPORTS ---
from typing import Tuple, List, Dict, Callable

import numpy as np
import pandas as pd
from icecream import ic
import polars as pl

from utils.statistics.metrics_helper import compute_VOTS_metrics


# - FUNCTIONS ---
def flatten_metadata_dict():
    pass


def flatten_metrics_dict(
    data: dict
) -> pd.DataFrame:
    # Flatten the data in a table
    flatten_data = []
    for frame, objects in data.items():
        for object_id, metrics in objects.items():
            flatten_object_info = {'frame': frame,
                                   'object_id': object_id}
            # TODO: actually the asme thing... 
            flatten_metrics = {f"{metric_name}": metric_value for metric_name, metric_value in metrics.items()}

            flatten_object_info.update(flatten_metrics)

            flatten_data.append(flatten_object_info)

    return pd.DataFrame(flatten_data)


def flatten_dataset_data_correlation(
    data: dict
) -> pd.DataFrame:
    flatten_data = []
    for sequence, objects in data.items():
        for object_id, coeffs in objects.items():
            flatten_data.append({
                'sequence': sequence,
                'object_id': object_id,
                'corr_Total_H_IoU': coeffs['corr_Total_H_IoU'],
                'corr_TP_H_IoU': coeffs['corr_TP_H_IoU'],
                'corr_TN_H_IoU': coeffs['corr_TN_H_IoU'],
                'corr_FP_H_IoU': coeffs['corr_FP_H_IoU'],
                'corr_FN_H_IoU': coeffs['corr_FN_H_IoU']
            })

    return pd.DataFrame(flatten_data)


def extract_obx_metrics(
    dataframe: pd.DataFrame,
    _obx: str
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, List[str]]:
    # [1:] to exclude the first frame as in DAVIS
    metrics = dataframe[dataframe['object_id'] == _obx][1:]

    # Even if target is not visible
    IoU_obx_curve = (metrics['IoU']).to_numpy()
    Total_H_curve = (metrics['Total_H']).to_numpy()
    TP_H_obx_curve = (metrics['TP_H']).to_numpy()
    TN_H_obx_curve = (metrics['TN_H']).to_numpy()
    FP_H_obx_curve = (metrics['FP_H']).to_numpy()
    FN_H_obx_curve = (metrics['FN_H']).to_numpy()

    yield IoU_obx_curve, Total_H_curve, TP_H_obx_curve, TN_H_obx_curve, \
        FP_H_obx_curve, FN_H_obx_curve

    VOTS_successes = (metrics['VOTS_successes']).to_numpy()
    VOTS_cases = list(metrics['VOTS_cases'])

    yield VOTS_successes, VOTS_cases


def save_dataframe_as_csv_table(
    df_data: dict,
    filename: str
):
    # save dataframe table as a CSV file
    df_data.to_csv(filename, index=False)


def save_dataframe_with_polars(
    df_data: dict,
    filename: str
):
    # save dataframe table as a CSV file
    df_data.to_csv(filename, index=False)
    polars_df = pl.from_pandas(df_data)
    polars_df.write_parquet(filename)
