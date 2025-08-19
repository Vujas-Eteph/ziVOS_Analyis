"""
Generate 3 talbes that list top-X% percentiles (Best/Worst/closest to 0)

Quick Use:
- streamlit run streamlit_apps/app_KPerformance_viewer.py 

by Stephane Vujasinovic
"""

# - IMPORTS ---
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import from parent folder
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path.cwd()))

import streamlit as st
from apps.streamlit_apps.st_configuration.configuration import streamlit_custom_page_configuration
from configuration.configuration_manager import ConfigManager


# - FUNCTION ---
# st.cache_data
def read_csv(
    df_path: str
) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame.
    """
    return pd.read_csv(df_path)


# st.cache_data
def extract_all_columns(
    df: pd.DataFrame
) -> list:
    """
    Extract columns of interest from the DataFrame.
    """
    return df.columns


# st.cache_data
def extract_columns_of_interest(
    df: pd.DataFrame
) -> list:
    """
    Extract columns of interest from the DataFrame.
    """
    return df.columns[2:-2]


# st.cache_data
def filter_by_coefficient(
    coeff_type: str,
    df: pd.DataFrame
):
    """
    Filter DataFrame based on a coefficient type for correlation and p-values.
    """
    corr_df = df[df["Coeff"] == f"{coeff_type}_corr_coeffs"]
    p_value_df = df[df["Coeff"] == f"{coeff_type}_p_values"]
    return corr_df, p_value_df


# st.cache_data
def filter_by_columns(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    """
    Select specific columns from the DataFrame.
    """
    return df[columns]


# st.cache_data
def convert_percentage_to_K(
    percentage: float,
    df: pd.DataFrame
) -> int:
    """
    Compute the index for the top K percentage of elements in the DataFrame.
    """
    index_count = int(percentage/100 * len(df))
    return index_count


def extract_top_and_worst_K(
    df: pd.DataFrame,
    sort_by: str,
    k_count: int
):
    """
    Sort DataFrame and select top and bottom K elements.
    """
    sorted_df = df.sort_values(by=sort_by)
    top_k = sorted_df.head(k_count)
    bottom_k = sorted_df.tail(k_count)
    return top_k, bottom_k


def extract_nearest_K_to_zero(
    df: pd.DataFrame,
    sort_by: str,
    columns: list,
    k_count: int
) -> pd.DataFrame:
    """
    Select elements nearest to zero based on a specific column.
    """
    df_filtered = df[columns[2:-2]].copy()
    df_filtered_squared = np.square(df_filtered)
    df["temp_sorting_column"] = df_filtered_squared[sort_by]
    sorted_df = df.sort_values(by="temp_sorting_column")
    nearest_k = sorted_df.head(k_count)
    return nearest_k.drop(columns=["temp_sorting_column"])


def compute_K_TMB(coeff_type, percentage, df, all_columns, column_to_sort_by):
    corr_df, p_value_df = filter_by_coefficient(coeff_type, df)
    filtered_corr_df = filter_by_columns(corr_df, all_columns)
    filtered_p_value_df = filter_by_columns(p_value_df, all_columns)
    k_elements_count = convert_percentage_to_K(percentage, filtered_corr_df)
    top_k, bottom_k = extract_top_and_worst_K(filtered_corr_df,
                                              column_to_sort_by,
                                              k_elements_count)
    nearest_k = extract_nearest_K_to_zero(filtered_corr_df, column_to_sort_by,
                                          all_columns,
                                          k_elements_count)

    dataframes = [top_k, nearest_k, bottom_k]
    columns_to_select = ["sequence_name", "obx", column_to_sort_by]
    top_k, nearest_k, bottom_k = [df[columns_to_select] for df in dataframes]

    return top_k, nearest_k, bottom_k


def gen_location_for_the_csv_file(_save_analysis_loc, mask_H=None):
    _save_analysis_loc = os.path.join(_save_analysis_loc, "raw_stats_HUB")
    if mask_H is not None:
        _save_analysis_loc = os.path.join(_save_analysis_loc, 
                                          f"mask_H_{mask_H}")
    else:
        _save_analysis_loc = os.path.join(_save_analysis_loc, "no_mask_H")

    return _save_analysis_loc


def display_table_with_pyplot(df):
    fig, ax = plt.subplots(1, 1, figsize=(1, 1))  # Adjust the size as needed
    # Hide axes
    ax.axis('off')
    # Table from DataFrame
    table = ax.table(cellText= df.values, colLabels=df.columns,
                     rowLabels=df.index, loc='center', cellLoc='center')
    # Make the text bold
    for (i, j), cell in table.get_celld().items():
        cell.set_text_props(fontweight='bold')
        # Auto-set the column widths to make the content fit well
    table.auto_set_column_width(col=list(range(len(df.columns))))
    st.pyplot(fig)


def process_and_display_data(
    coeff_type: str,
    df_paths: str,
    percentage: float,
):
    """
    Main process function to read, filter, and display data.
    """
    df_0 = read_csv(df_paths[0])
    all_columns = extract_all_columns(df_0)
    columns_of_interest = extract_columns_of_interest(df_0)
    with st.sidebar:
        column_to_sort_by = st.multiselect("Entropy type", columns_of_interest,
                                           key="column_to_sort_by")

    for entropy_type in column_to_sort_by:
        st.header(f":{ST_COLOR}[{entropy_type}]", divider=ST_COLOR)
        for df_path in df_paths:
            df = read_csv(df_path)
            top_K_df, mid_K_df, low_K_df = compute_K_TMB(coeff_type,
                                                        percentage,
                                                        df, all_columns,
                                                        entropy_type)

            st.write(f"#### :{ST_COLOR}[{df_path}]")
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.subheader(f":{ST_COLOR}[Top-{percentage}%]")
                st.dataframe(top_K_df, width=1000)
                display_table_with_pyplot(top_K_df)
            with col_2:
                st.subheader(f":{ST_COLOR}[{percentage}% Nearest to 0]")
                st.dataframe(mid_K_df, width=1000)
                display_table_with_pyplot(mid_K_df)
            with col_3:
                st.subheader(f":{ST_COLOR}[Low-{percentage}%]")
                st.dataframe(low_K_df, width=1000)
                display_table_with_pyplot(low_K_df)


# - MAIN --
if __name__ == '__main__':
    # Configuration/ Available methods and datasets
    config = ConfigManager()
    avail_datasets = config.get_all_available_datasets()
    avail_methods = config.get_all_available_methods()

    ST_COLOR = 'green'
    
    streamlit_custom_page_configuration("Summary Statistics Viewer",
                                        ":eyeglasses:")
    METHOD_OPTIONS = avail_methods
    DATASET_OPTIONS = avail_datasets
    COEFF_OPTIONS = ["spearman", "pearson"]
    st.header("", divider=ST_COLOR)
    with st.sidebar:
        st.subheader(f":{ST_COLOR}[Control Flow]", divider=ST_COLOR)
        method_name = st.selectbox("Method", METHOD_OPTIONS, key="method")
        dataset_name = st.selectbox("Dataset", DATASET_OPTIONS, key="dataset")
        with st.form("flow_form"):
            sim_or_not = st.multiselect(('sim_or_not'),
                                        options=["noxsim", "yexsim"],
                                        default=[])
            sum_median_mean = st.multiselect(('mum_median_mean'),
                                             options=["sum", "mean"],
                                             default=[])
            uncouple_wrt_to = st.multiselect(('uncouple_wrt_to'),
                                             options=["base", "gt", "pd"],
                                             default=[])
            submitted = st.form_submit_button("Submit")

        # Prep. configuration
        config['dataset_name'] = dataset_name
        config['method_name'] = method_name
        config_generator = config.get_my_configuration()

        # TODO: adapt this !!
        image_directory, gt_mask_directory = next(config_generator)
        pred_mask_directory, logits_directory, softmax_directory = \
            next(config_generator)
        entropy_directory, confusion_directory = next(config_generator)
        preprocessing_directory = next(config_generator)

        save_analysis_loc = config.get_iou_entropy_analysis_path()
        loc_to_save_csv_results = gen_location_for_the_csv_file(save_analysis_loc, 5)

        raw_stats_path = loc_to_save_csv_results
        root, end = raw_stats_path.split('/raw_stats_HUB/')
        final_stats_path = os.path.join(root, "summary_stats", end)
        stats_location = os.path.join(final_stats_path)
        directories = os.listdir(stats_location)

        check = lambda file_name, options: any(o in file_name for o in options)

        if sim_or_not != []:
            directories = [f for f in directories if check(f, sim_or_not)]
        if sum_median_mean != []:
            directories = [f for f in directories if check(f, sum_median_mean)]
        if uncouple_wrt_to != []:
            directories = [f for f in directories if check(f, uncouple_wrt_to)]

        csv_files_to_read = st.multiselect(('csv files'), options=directories)
        csv_paths = \
            [os.path.join(stats_location, f) for f in csv_files_to_read]
        coeff_type = st.selectbox("Correlation Coefficient",
                                  COEFF_OPTIONS, key="coeff_type")
        percentage = st.slider("percentage %",
                               min_value=0, max_value=50, value=10)

    print(csv_paths)
    if csv_paths != []:
        process_and_display_data(coeff_type, csv_paths, percentage)
