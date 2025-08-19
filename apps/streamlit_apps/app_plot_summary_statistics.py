"""
Display summary statistics in graphics.
Also allows for seeing which attributes are attached to which obx and sqx.

Quick Use
- streamlit run streamlit_apps/app_plot_summary_statistics.py 

by Stéphane Vujasinovic
"""

# - IMPORTS ---
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Import from parent folder
import sys
from pathlib import Path
import yaml

# Add the parent directory to sys.path
sys.path.append(str(Path.cwd()))

import streamlit as st
from apps.streamlit_apps.st_configuration.configuration import streamlit_custom_page_configuration
from apps.streamlit_apps.st_utils.st_color_map import get_color_dict
from configuration.configuration_manager import ConfigManager

from app_KPerformance_viewer import gen_location_for_the_csv_file

# - CONSTANTS ---
ATTRIBUTES_EXPLICATION = {'BC': 'Background Clutter',
                          'DEF': 'Deformation',
                          'MB': 'Motion Blur',
                          'FM': 'Fast-Motion',
                          'LR': 'Low Resolution',
                          'OCC': 'Occlusion',
                          'OV': 'Out-of-view',
                          'SV': 'Scale-Variation',
                          'AC': 'Appearance Change',
                          'EA': 'Edge Ambiguity',
                          'CS': 'Camera-Shake',
                          'HO': 'Heterogeneous Object',
                          'IO': 'Interacting Objects',
                          'DB': 'Dynamic Background',
                          'SC': 'Shape Complexity',
                          'POCC': 'Partial Occlusion',
                          'FOCC': 'Full Occlusion',
                          'COCC': 'Cross-Occlusion',
                          'VLR': 'Very Low Resolution',
                          'DIS': 'Distractors',
                          'ROT': 'In plane Rotation'
                          }
SUFFIXES = [" :::: ",  # Needed as graduations are not available with streamlit
            " ::::::::: ",
            " ::::::::: ",
            " ::::::::: ",
            " :::::::: ",
            " :::::::: ",
            " ::::::::: ",
            " ::::::::: ",
            " :::::::: ",
            " ::::::::: ",
            " ::::::::: ",
            " :::::::: ",
            " ::::::::: ",
            " :::::::: ",
            " :::::::: ",
            " ::::::: ",
            " ::::::: ",
            " ::::::: ",
            " ::::::: ",
            " ::::::: ",
            " ::::::: "
            ]


# - MAIN ---
if __name__ == '__main__':
    # Configuration/ Available methods and datasets
    config = ConfigManager()
    avail_datasets = config.get_all_available_datasets()
    avail_methods = config.get_all_available_methods()
    
    ST_COLOR = 'orange'

    streamlit_custom_page_configuration("Display statistics", ":bar_chart:")
    METHOD_OPTIONS = avail_methods
    DATASET_OPTIONS = avail_datasets
    
    colors = get_color_dict()
    st.header("", divider=ST_COLOR)
    with st.sidebar:
        st.subheader(f":{ST_COLOR}[Control Flow]", divider=ST_COLOR)
        method_name = st.selectbox("Method", METHOD_OPTIONS, key="method")
        dataset_name = st.selectbox("Dataset", DATASET_OPTIONS, key="dataset")
        
        st.subheader(f":{ST_COLOR}[Graph Layout]", divider=ST_COLOR)
        blob_size = st.slider(label="blob_size", min_value=10, max_value=30,
                              value=20)
        background_blob_opacity = st.slider(label="background_blob_opacity",
                                            min_value=0.0, max_value=1.0,
                                            value=0.5)

        st.subheader(f":{ST_COLOR}[Control Flow]", divider=ST_COLOR)
        sim_or_not = st.multiselect(('sim_or_not'),
                                    options=["noxsim", "yexsim"],
                                    default=[])
        sum_median_mean = st.multiselect(('mum_median_mean'),
                                         options=["sum", "mean"],
                                         default=[])
        uncouple_wrt_to = st.multiselect(('uncouple_wrt_to'),
                                         options=["base", "gt", "pd"],
                                         default=[])

        # Prep. configuration
        config['dataset_name'] = dataset_name
        config['method_name'] = method_name
        config_generator = config.get_my_configuration()

        # TODO: Also replace this one...
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
        all_csv_files = \
            [os.path.join(stats_location, f) for f in csv_files_to_read]

        if all_csv_files != []:
            df_table = pd.read_csv(all_csv_files[0])

            coeff_oI = st.selectbox("coeff_oI",
                                    options=[e for e in list(set(df_table["Coeff"])) if "coeffs" in e])

            entropy_options = df_table.columns[2:-2]
            entropy_oI = st.multiselect(('Available entropies'),
                                        options=entropy_options)
            selected_H = st.multiselect(('Focus on these entropies'),
                                        options=entropy_oI)

    # If DAVIS, then read the attributes of the dataset
    attribute_oI = False
    if dataset_name == "d17-val":  # TODO: Also adapt for LVOS
        path_to_attributes = os.path.join("data", "attributes", dataset_name,
                                          "Unofficial_attributes.csv")
        attributed_table = pd.read_csv(path_to_attributes)
        options = [False]
        options += list(attributed_table.columns[1:])
        attribute_oI = st.select_slider("Attributes", options=options)
        display_graduations = ""
        for o, suffix in zip(options, SUFFIXES):
            if o == attribute_oI:
                display_graduations += f":{ST_COLOR}[{o}]" + suffix
                continue
            display_graduations += f"{o}" + suffix
        display_graduations = display_graduations.rstrip(suffix)
        
        st.subheader(display_graduations)
        if attribute_oI != False:
            st.subheader(f"Selected Attributes :{ST_COLOR}[{attribute_oI} - "
                         f"{ATTRIBUTES_EXPLICATION[attribute_oI]}]")
    elif dataset_name == "lvos-val":
        options = [False]

    for csv_file in all_csv_files:
        st.header(f":{ST_COLOR}[{csv_file}]", divider=ST_COLOR)
        df_table = pd.read_csv(csv_file)
        
        # Order the values from hight to lowest values    # Save the image if button pressed
        sort_by = st.selectbox("sort_by",
                               options=["False"] + selected_H)
        
        all_coeff_types = np.unique(df_table["Coeff"])

        fig = go.Figure(layout=go.Layout(title=csv_file,
                                         font=dict(size=18)))
        for coeff_type in all_coeff_types:
            if coeff_type not in coeff_oI:
                continue

            table = df_table[df_table["Coeff"] == coeff_type].copy()
            table_p_value = df_table[df_table["Coeff"] == 'spearman_p_values'].copy()
            entropy_type_titles = \
                [col for col in table.columns if col in entropy_oI]
                
            table["obx"] = table["obx"].str.replace("object", "", regex=False)
            table['sequence_name'] = table['sequence_name'] + table["obx"]

            if attribute_oI != False:
                # Find rows where the attribute is equal to 1
                filter = list(attributed_table[attribute_oI] == 1)

                # Select the 'Sequence Name' column
                sequence_names = attributed_table['Sequence Name']
                sequence_with_oI_attribute = attributed_table[filter]['Sequence Name']
                table = table[table['sequence_name'].isin(sequence_with_oI_attribute)]
            
        
        table = table.reset_index()
        if sort_by != "False":
            table = table.sort_values(by=sort_by)

        for h_name in selected_H:
            fig.add_trace(
                go.Scatter(
                    x=table["sequence_name"],
                    y=table[h_name],
                    # line_dash="dash",
                    # mode='markers',
                    marker=dict(color=colors[h_name]),
                    name=h_name
                )
            )

        fig.add_hline(y=0,
                        line_width=3,
                        line_dash="dash",
                        line_color="green")

        fig.update_traces(marker=dict(size=blob_size, opacity=0.5),
                            selector=dict(mode='markers'))

        # Place the legend on the right-up corner of the graph
        fig.update_layout(legend=dict(orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1),
                            yaxis_range=[-1.02, 1.02])
        st.plotly_chart(fig, theme=None, use_container_width=True)
            
            
    # Save the image if button pressed
    if st.button("Save graphic", type="primary"):
        st.write("Graph saved")
        fig.write_image("fig1.png")
    else:
        pass
    
    # Save the data as a csv file
    if st.toggle("Save data as csv"):
        title = st.text_input('csv title', '')
        # st.write(table)
        table.rename(columns={table.columns[0]: 'id'}, inplace=True)
        table_to_save = table[["id", "sequence_name"] + selected_H].copy()
        table_to_save["id"] = table.index.values
        # Check columns names, and replace "_" with " ". # Need to get rid of "_" to work with latex
        table_to_save["sequence_name"] = table_to_save["sequence_name"].str.replace("_", " ", regex=False)
        new_columns_name = [col.replace("_", " ") for col in table_to_save.columns]
        latex_rename = {}
        for col in table_to_save.columns:
            new_col = col.replace("_", " ")
            latex_rename[col] = new_col
        table_to_save.rename(columns=latex_rename, inplace=True)
        st.write(table_to_save)

        # st.write(table_to_save)
        if st.button("Save data as csv", type="primary"):
            st.write("csv file created saved")
            if title == "":
                st.write("No name is given, saving to 'no_name.csv")
                title = "no_name.csv"
            table_to_save.to_csv(title, sep=',', index=False, encoding='utf-8')
    else:
        pass

