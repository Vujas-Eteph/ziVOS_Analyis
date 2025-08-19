'''
Compute summary statistics #TODO Give more details

by Stéphane Vujasinovic
'''

# - IMPORTS ---
import os
import argparse
import numpy as np
import polars as pl
import pandas as pd
from icecream import ic
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import itertools
import copy


from utils.path_utils import create_directory_if_not_there

from configuration.configuration_manager import ConfigManager

from analyze_iou_vs_entropy import gen_location_for_the_csv_file

import shutil
import warnings


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
    parser.add_argument("--warning", action="store_true",
                        help="Add icecream statements")
    parser.add_argument("-T", "--temperature", type=float, default=1.0,
                        help="Apply temperature scaling to the softmax output")

    return parser.parse_args()


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


def uncouple_dataframe_from_object_size(
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
    if _use_object_size == 'gt':
        obj_sizes = curves_of_interest["gt_object_size"]
    elif _use_object_size == 'pd':
        obj_sizes = curves_of_interest["pd_object_size"]
    # check if obj_size has 0.
    for key in curves_to_div_by_objet_size.keys():
        curves_uncorrelated_to_object_size = \
            np.array(curves_to_div_by_objet_size[key]/obj_sizes)
        # Deal with nans and infs
        curves_uncorrelated_to_object_size[np.isnan(curves_uncorrelated_to_object_size)] = np.nan
        curves_uncorrelated_to_object_size[curves_uncorrelated_to_object_size == np.inf] = np.nan
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

    # Compute the mean results
    prefix = "Total"
    mean_Total_curves = {key: curves_to_mean[key]/curves_of_interest["Image_size"] for key in curves_to_mean.keys() if prefix in key}
    prefix = "TP"
    mean_TP_curves = {key: curves_to_mean[key]/curves_of_interest["TP_size"] for key in curves_to_mean.keys() if prefix in key}
    prefix = "TN"
    mean_TN_curves = {key: curves_to_mean[key]/curves_of_interest["TN_size"] for key in curves_to_mean.keys() if prefix in key}
    prefix = "FP"
    mean_FP_curves = {key: curves_to_mean[key]/curves_of_interest["FP_size"] for key in curves_to_mean.keys() if prefix in key}
    prefix = "FN"
    mean_FN_curves = {key: curves_to_mean[key]/curves_of_interest["FN_size"] for key in curves_to_mean.keys() if prefix in key}
    prefix = "TR"
    mean_TR_curves = {key: curves_to_mean[key]/curves_of_interest["TR_size"] for key in curves_to_mean.keys() if prefix in key}
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


def extract_curves(
    _df_obx
) -> dict:
    EXCLUDE = ["frame", "object_id", "Masked_value"]
    columns = _df_obx.columns
    columns_of_interest = list(set(columns) - set(EXCLUDE))

    curves_of_interest = {coi: _df_obx[coi] for coi in columns_of_interest}

    return curves_of_interest


def create_table_titles(
    _df_table
):
    _columns = ["Coeff"] + list(_df_table.columns)[1:]
    return pd.DataFrame(columns=_columns)


# Load the csv files by groupes
def extract_seq_table(_dataset_path, _sequence_name, _key):
    df_table_seq = None
    objects_in_sequence = os.listdir(os.path.join(_dataset_path,
                                                  _sequence_name))

    for odx, object_id in enumerate(objects_in_sequence):
        csv_file = os.path.join(_dataset_path, _sequence_name, object_id,
                                f"{_key}.csv")
        df_table = pd.read_csv(csv_file)
        
        if odx == 0:
            df_table_seq = create_table_titles(df_table)

        # Read the csv file
        if df_table_seq is not None:
            df_table_seq = pd.concat([df_table_seq, df_table],
                                     ignore_index=True)
        else:
            df_table_seq = df_table

    return df_table_seq


# Loop over the different coefficients
def extract_table(dataset_path, k, v):
    for sdx, sequence_name in enumerate(v):
        if sdx == 0:
            concat_df_table = extract_seq_table(dataset_path, sequence_name, k)
        else:
            _df_table = extract_seq_table(dataset_path, sequence_name, k)
            concat_df_table = concat_df_table._append(_df_table,
                                                      ignore_index=True)

    # Assign the values from 'Unnamed: 0' to 'Coeff'
    concat_df_table["Coeff"] = concat_df_table["Unnamed: 0"]
    concat_df_table.drop(columns=["Unnamed: 0"], inplace=True)
    return concat_df_table


def derive(entropy_curves):
    out = []
    for entropy_curve in entropy_curves:
        function_decaler = entropy_curve[1:]
        function_decaler = np.append(function_decaler,
                                    np.nan)
        derivative = function_decaler - entropy_curve
        derivative[0] = np.nan
        out.append(derivative)

    return out


# - MAIN ---
if __name__ == "__main__":
    args = arguments_parser()
    dataset_name = args.dataset
    method_name = args.method
    #
    # Useful for debugging
    if args.verbose:
        ic.disable()
    if not args.warning:
        warnings.filterwarnings("ignore", category=Warning)

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
    
    if args.temperature != 1.0:
        entropy_directory = entropy_directory + f'_Temp_{args.temperature}'
        loc_to_save_csv_results += f'_Temp_{args.temperature}'

    raw_stats_path = loc_to_save_csv_results
    root, _ = raw_stats_path.split('/raw_stats_HUB/')
    end = f"mask_H_{args.mask_H}"
    intermediate_stats_path = os.path.join(root, "intermediate_stats", end)
    final_stats_path = os.path.join(root, "summary_stats", end)

    sim_options = ["noxsim", "yexsim"]  # 'yexsim_', 'noxsim_'
    operation_options = ["sum", "mean"]  # 'mean', 'median', 'sum'
    use_object_options = ["base", "pd"]  # 'base', 'gt', 'pd'

    parameters = [sim_options, operation_options, use_object_options]

    all_parquet_files = [e for e in os.listdir(raw_stats_path)
                         if os.path.isfile(os.path.join(raw_stats_path, e))]
    all_parquet_files = sorted(all_parquet_files)

    print(f"Extracting summary statistics for each sequence and object in: {all_parquet_files}")
    # Read every parquet file
    for parquet_file in tqdm(all_parquet_files):
        sequence_name = parquet_file.split('.')[0]
        # ic.enable()
        # ic(sequence_name)
        # LIST = ["7BcOR5aJ"]
        # if sequence_name in LIST:
        #     continue
        # ic.disable()
        parquet_file_loc = os.path.join(raw_stats_path, parquet_file)
        df = pl.read_parquet(parquet_file_loc)

        # For each object compute the correlation coefficients
        for obj_id in set(df["object_id"]):
            filtered_df = df.filter(df["object_id"] == obj_id)
            object_info_dict = {}
            for (use_sim, operation_type,
                    use_object_size) in itertools.product(*parameters):

                curves = extract_curves(filtered_df).copy()
                if use_sim == "yexsim":
                    curves = adapt_dataframe_with_simulation(curves)
                if operation_type == "mean":
                    curves = adapt_dataframe_with_mean_operation(curves)
                    if use_object_size != "base":
                        continue
                if operation_type == "sum":
                    if use_object_size != "base":
                        curves = \
                            uncouple_dataframe_from_object_size(curves,
                                                                use_object_size)

                performance_curves = extract_performance_related_curves(curves)
                if method_name == "QDMN":
                    QAM_curves = dict()
                    CURVES_OF_INTEREST = ["QAM_score_fdx", "QAM_score_fdx_obx"]
                    QAM_curves.update({key_oI: curves[key_oI] for key_oI in CURVES_OF_INTEREST})
                    entropy_curves = QAM_curves
                else:
                    entropy_curves = extract_entropy_related_curves(curves)
                presence_curves = extract_presence_related_curves(curves)
                

                # derivatives = derive(entropy_curves)      # TODO: adapt
                # derivative_curves[curve_key] = derivative

                # shift the performance (only one curve in comparision to entropy curves)
                IoU_curve = np.array(performance_curves["IoU"])

                spearman_corr_coeffs = {}
                spearman_p_values = {}
                # pearson_corr_coeffs = {}
                # pearson_p_values = {}
                # - Loop over the curves
                for curve_key, _ in entropy_curves.items():
                    # Warning ! Discarding the fist frame, similarly to DAVIS 2017
                    S_correlation, p_value = \
                        spearmanr(IoU_curve[1:],
                                  entropy_curves[curve_key][1:],
                                  nan_policy='omit')
                    spearman_corr_coeffs.update({curve_key: S_correlation})
                    spearman_p_values.update({curve_key: p_value})
                    # if shift == 0:  # as pearson does not work with nans
                    #     S_correlation, p_value = \
                    #         pearsonr(IoU_curve[1:],
                    #                  entropy_curves[curve_key][1:])
                    #     pearson_corr_coeffs.update({curve_key: S_correlation})
                    #     pearson_p_values.update({curve_key: p_value})
                    # Save in an object dict
                    object_info_key = f"{use_sim}_{operation_type}_{use_object_size}"
                    object_info_value = dict()
                    object_info_value.update({"spearman_corr_coeffs": spearman_corr_coeffs})
                    object_info_value.update({"spearman_p_values": spearman_p_values})
                    # if shift == 0:
                    #     object_info_value.update({"pearson_corr_coeffs": pearson_corr_coeffs})
                    #     object_info_value.update({"pearson_p_values": pearson_p_values})
                    object_info_dict[object_info_key] = object_info_value

                flatten_data = []
                for k, v in object_info_dict.items():
                    name_for_the_csv_file = k
                    path_to_save_results = \
                        os.path.join(intermediate_stats_path,
                                        sequence_name, obj_id)
                    create_directory_if_not_there(path_to_save_results)
                    csv_file_name = os.path.join(path_to_save_results,
                                                    f"{name_for_the_csv_file}.csv")
                    df_table = pd.DataFrame(v).T
                    df_table = df_table.assign(sequence_name=sequence_name,
                                                obx=obj_id)
                    df_table.to_csv(csv_file_name)

    dataset_path = os.path.join(intermediate_stats_path)
    all_csv_dir = [
        os.path.join(
            dataset_path, filename, "object_1"
        )  # There is always at least the first object present
        for filename in os.listdir(
            dataset_path
        )  # Iterate through each file in the directory
        if os.path.isdir(
            os.path.join(dataset_path, filename)
        )  # Check if the item is a file
    ]

    all_csv_dir = sorted(all_csv_dir)

    # - Save in a dict, the files names, and the sequences names
    # to better read the csv file later and work with them
    # list add csv files in a sequence
    print("Re-Collecting all stastitics")
    for idx, csv_dir in tqdm(enumerate(all_csv_dir)):
        sequence_name = csv_dir.split("/")[-2]  # when using windows replace "/" with "\\"
        all_csv_files = os.listdir(csv_dir)
        if idx == 0:
            csv_files_dict = dict()
            for csv_files in all_csv_files:
                new_key = csv_files.split(".")[0]
                csv_files_dict[new_key] = [sequence_name]
        else:
            for csv_files in all_csv_files:
                key = csv_files.split(".")[0]
                value = csv_files_dict[key] + [sequence_name]
                csv_files_dict.update({key: value})

    print("Writing a single csv file")
    for k, v in tqdm(csv_files_dict.items()):
        # Read the corresponding file
        df_table = extract_table(dataset_path, k, v)

        # Save the resulting dataframe in a csv file
        path_to_save_results = os.path.join(final_stats_path)
        create_directory_if_not_there(path_to_save_results)
        file_loc = os.path.join(path_to_save_results, f"{k}.csv")
        df_table.to_csv(file_loc)

    # Delete the intermediate_stats_path
    shutil.rmtree(os.path.join(root, "intermediate_stats"))
