"""
Read the benchmark results and display the results - Graph and summary stats.

Quick Use
- streamlit run streamlit_apps/app_benchmark.py 

by Stéphane Vujasinovic
"""

# - IMPORTS ---
import os
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
# Import from parent folder
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path.cwd()))

import streamlit as st
from apps.streamlit_apps.st_configuration.configuration import streamlit_custom_page_configuration
from apps.streamlit_apps.st_utils.st_color_map import get_color_dict

from configuration.configuration_manager import ConfigManager


# - FUNCTIONS ---


# - MAIN ---
if __name__ == '__main__':
    # Configuration/ Available methods and datasets
    config = ConfigManager()
    avail_datasets = config.get_all_available_datasets()
    avail_methods = config.get_all_available_methods()

    METHOD_OPTIONS = avail_methods
    DATASET_OPTIONS = avail_datasets

    METRICS_OPTIONS = ["J&F", "J", "F"]
    ST_COLOR = 'blue'
    OPACITY = 0.5

    streamlit_custom_page_configuration("Benchmark Results", ":sunglasses:")
    colors = get_color_dict()
    st.header("", divider=ST_COLOR)
    st.header("", divider=ST_COLOR)
    st.header(f":{ST_COLOR}[sVOS Benchmark] - Official Results")

    with st.sidebar:
        dataset_name = st.multiselect(('Dataset Option'),
                                      options=DATASET_OPTIONS,
                                      default=["d17-val"])
        methods = st.multiselect(('Method Option'),
                                 options=METHOD_OPTIONS,
                                 default=["XMem"])

    # Prep. configuration
    config['dataset_name'] = dataset_name[0]
    app_col = st.columns(len(methods))
    fig = go.Figure(layout=go.Layout(title=dataset_name[0],
                                     font=dict(size=18),
                                     yaxis=dict(range=[0, 102])))
    colors = ["#ff4b4b", "#4b4bff", "#4bff4b", "#ffff4b", "#ff4bff", "#4b4b00",
              "#400f4b", "#005f00", "#ff4b00", "#00ffff", "#ffffff"]
    method_used = []
    name_id = []
    J_and_F = []
    J = []
    F = []
    markdown_table = dict()
    for method_name, st_col, color in zip(methods, app_col, colors):
        with st_col:
            config['method_name'] = method_name
            config_generator = config.get_my_configuration()
            _, _ = next(config_generator)
            benchmark_results_dir, csv_file_name = \
                config.get_benchmark_dir_location()
            csv_benchmark_results = os.path.join(benchmark_results_dir,
                                                 csv_file_name)
            st.subheader(f'Table for {csv_benchmark_results}\n')
            df = pd.read_csv(csv_benchmark_results)
            # st.write(df)
            st.subheader(f":{ST_COLOR}[{method_name}] Performance:")
            st.write(df)
            global_J_and_F = df[df.columns[0]][0]
            st.subheader(f"{df.columns[2]}: :{ST_COLOR}[{global_J_and_F}]")
            global_J = df[df.columns[1]][0]
            st.subheader(f"{df.columns[3]}: :{ST_COLOR}[{global_J}]")
            global_F = df[df.columns[4]][0]
            st.subheader(f"{df.columns[4]}: :{ST_COLOR}[{global_F}]")

            method_used += [method_name for _ in df[df.columns[0]][1:]]
            name_id += [a.rstrip(' ') + '_' + b for a, b in zip(df[df.columns[0]][1:], df[df.columns[1]][1:])]
            J_and_F += list(df[df.columns[2]][1:])
            J += list(df[df.columns[3]][1:])
            F += list(df[df.columns[4]][1:])

            markdown_table[method_name] = {"J&F":global_J_and_F,
                                           "J":global_J,
                                           "F":global_F}

    _col_1, _col_2 = st.columns(2)
    with _col_1:
        st.subheader(f" - :{ST_COLOR}[Performance] Table")
        st.text(pd.DataFrame(markdown_table).T.to_markdown())

        performance_dict = {"method name": method_used,
                            "sequence_obx_id": name_id,
                            "J&F": J_and_F,
                            "J": J,
                            "F": F}

    metric = st.selectbox(('Metric'), options=METRICS_OPTIONS)

    fig = px.histogram(performance_dict,
                       x="sequence_obx_id",
                       y=metric,
                       color="method name",
                       opacity=OPACITY,
                       barmode="group",
                       color_discrete_sequence=colors,
                       title=dataset_name[0])
    # Legend
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                  y=1.02, xanchor="right", x=1))
    # y-axis name
    fig.update_layout(yaxis_title=f"{metric}")
    st.plotly_chart(fig, theme=None, use_container_width=True)

    # # Legend
    # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
    #                               xanchor="right", x=1))

    # st.plotly_chart(fig, theme=None, use_container_width=True)
    
    # Discard elements?
    for mdx, (method_name, color) in enumerate(zip(methods, colors)):
        FF = dict()
        N = []
        G = []
        method_used = []
        name_id = []
        count = []
        if method_name == "XMem":  # skip as no prompts isseud for the baseline
            continue

        config['method_name'] = method_name
        config_generator = config.get_my_configuration()
        _, _ = next(config_generator)
        _, _, _ = next(config_generator)
        T = config.get_prompts_history_dir_location()
        O = os.listdir(T)
        O = [os.path.join(T,o) for o in O if os.path.isdir(os.path.join(T, o))]
        all_prompt_parquet = [os.path.join(o, s) for o in O for s in os.listdir(o)]


        with st.sidebar:
            options = [elem.split("/")[-2:] for elem in all_prompt_parquet]
            for option in options:
                option[1] = option[1].rstrip(".parquet").lstrip("id_").zfill(3)
            options = ["_".join(elem) for elem in options]
            discarded_obx = st.multiselect('Discard the following elements',
                                            options=options,
                                            default=[])
        break

    st.header("", divider=ST_COLOR)
    st.header("", divider=ST_COLOR)
    st.header(f":{ST_COLOR}[Interaction Benchmark]")
    markdown_table = dict()
    for mdx, (method_name, color) in enumerate(zip(methods, colors)):
        FF = dict()
        VV = dict()
        Delat_P_cumulative_count_curve = dict()
        method_used = []
        name_id = []
        count = []
        if method_name == "XMem":  # skip as no prompts isseud for the baseline
            continue

        config['method_name'] = method_name
        config_generator = config.get_my_configuration()
        _, _ = next(config_generator)
        _, _, _ = next(config_generator)
        T = config.get_prompts_history_dir_location()
        O = os.listdir(T)
        O = [os.path.join(T, o) for o in O if os.path.isdir(os.path.join(T, o))]
        all_prompt_parquet = [os.path.join(o, s) for o in O for s in os.listdir(o)]

        for p_obx in all_prompt_parquet:
            seq_name, obj_id = p_obx.split("/")[-2:]
            obj_id = obj_id.rstrip(".parquet").lstrip("id_")
            identifier = f"{seq_name}_{obj_id.zfill(3)}"
            if identifier in discarded_obx:
                st.write(f"Discarded: {identifier}")
                continue
            method_used.append(method_name)
            name_id.append(identifier)
            data = pl.read_parquet(p_obx)
            # Count number of prompts
            count.append((data["prompt"].sum()))
            filter = np.array(data["prompt"]).astype(bool)
            fdx_w_prompt = np.array(data["fdx"])[filter]
            total_number_of_frames = (data["fdx"])[-1] + 1
            IoU_variations = np.array(data["IoU"])
            Entropy_variations = np.array(data["H"])
            # Add the first and the last frame as interecation after the count
            fdx_w_prompt = np.concatenate((np.array([0]),fdx_w_prompt), axis=0)
            fdx_w_prompt = np.concatenate((fdx_w_prompt, np.array([data["fdx"][-1]])), axis=0)
            l1_dist = fdx_w_prompt[1:] - fdx_w_prompt[:-1]  # if equal to 0, then one interaction after another
            number_of_consecutive_prompts = (l1_dist == 1).sum()
            number_of_consecutive_prompts = number_of_consecutive_prompts + 1 if number_of_consecutive_prompts != 0 else number_of_consecutive_prompts
            FF[identifier] = l1_dist
            VV[identifier] = number_of_consecutive_prompts
            print(np.sort(l1_dist))
            unique, counts = np.unique(l1_dist, return_counts=True)
            c_j = dict(zip(unique,counts))
            print(c_j)
            print(total_number_of_frames)
            print(c_j.keys())
            n_curve = [0]
            c_j_keys = c_j.keys()
            for i in range(1,total_number_of_frames+1):
                new_n_curve_value = n_curve[-1]
                if i in c_j_keys:
                    new_n_curve_value += c_j[i]
                n_curve.append(new_n_curve_value)
                
            print(n_curve)
            print(np.sum(np.array(n_curve)))
            print(total_number_of_frames)
            R = np.sum(np.array(n_curve))/total_number_of_frames
            print(R)
            print(identifier)
            Delat_P_cumulative_count_curve[identifier] = R

        # Convert to dataframe
        method_prompt_stats_dict = {"method name": method_used,
                                    "sequence_obx_id": name_id,
                                    "prompt_count": count}
        method_prompt_stats_df = pd.DataFrame(method_prompt_stats_dict)
        fig = px.bar(method_prompt_stats_df,
                     x="sequence_obx_id",
                     y="prompt_count",
                     color="method name",
                     opacity=OPACITY,
                     color_discrete_sequence=[color],
                     title=dataset_name[0])

        # Legend
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                    y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, theme=None, use_container_width=True)
        col1, col2 = st.columns(2)
        
        delta_P_cummulative_count_AUC = []
        for k,v in Delat_P_cumulative_count_curve.items():
            delta_P_cummulative_count_AUC.append(v)
        
        delta_P_cummulative_count_AUC_array = np.array(delta_P_cummulative_count_AUC)
        delta_P_cummulative_count_AUC_array_mean = np.mean(delta_P_cummulative_count_AUC_array)
        
        # st.write(cdf_data)
        
        csv_filename = f"cdf_prompt_{method_name}.csv"
        
        import csv
        # with open(csv_filename, "w", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(["x", f"{dataset_name[0]}"])
        #     writer.writerows(cdf_data)

        # Calculate the area under the curve (AUC)
        #auc = np.trapz(cumulative_count, l1_distance_sorted)
        # st.write("Cumulative count")
        # st.write(cumulative_count)
        # st.write("l1_distance_sorted")
        # st.write(l1_distance_sorted)

        st.write("Delta Metric:", round(delta_P_cummulative_count_AUC_array_mean,3))        
        
        video_sequence_length_lvos = []

        with col1:
            st.write(method_prompt_stats_df)
        with col2:
            st.subheader("Custom metrics for iVOTS")
            st.subheader(f"Threshold used: :{ST_COLOR}[0.7] (fixed, need to read it from the metadata")
            total_number_of_prompts = sum(method_prompt_stats_dict['prompt_count'])
            prompt_array = np.array(method_prompt_stats_dict['prompt_count'])
            st.subheader(f":{ST_COLOR}[NoP] - {dataset_name[0]}: :{ST_COLOR}[{total_number_of_prompts}]")
            st.subheader(f":{ST_COLOR}[DeltaP] - {dataset_name[0]}: :{ST_COLOR}[{round(delta_P_cummulative_count_AUC_array_mean,3)}]")
            st.subheader(f":{ST_COLOR}[avg NOP] - {dataset_name[0]}: :{ST_COLOR}[{np.mean(prompt_array)}]")
            st.subheader(f":{ST_COLOR}[std NOP] - {dataset_name[0]}: :{ST_COLOR}[{np.std(prompt_array)}]")

        N = []
        G = []
        for k in FF.keys():
            v = FF[k]
            if v.size == 0:
                v = [0]
            for e in v:
                N.append(k)
                G.append(e)

        FF_bis = {"identifier": N,
                  "l1_value": G}
        FF_df = pd.DataFrame(FF_bis)
        
        fig = px.box(FF_df, x="identifier", y="l1_value", title=dataset_name[0])
        st.plotly_chart(fig, theme=None, use_container_width=True)

        # st.write(FF_df["l1_value"])
        l1_dist_vec = np.array(FF_df["l1_value"])
        l1_dist_mask = l1_dist_vec != 0
        l1_dist_avg = np.mean(l1_dist_vec[l1_dist_mask])
        
        N = []
        G = []
        for k in VV.keys():
            v = VV[k]
            N.append(k)
            G.append(v)
        VV_bis = {"identifier": N,
                  "consecutive_prompts": G}
        VV_df = pd.DataFrame(VV_bis)
        
        fig = px.bar(VV_df, x="identifier", y="consecutive_prompts", title=dataset_name[0])
        st.plotly_chart(fig, theme=None, use_container_width=True)
        st.header("", divider=ST_COLOR)
        
        # st.write(VV_df["consecutive_prompts"])
        cons_p_dist_vec = np.array(VV_df["consecutive_prompts"])
        cons_p_dist_sum = np.sum(cons_p_dist_vec)
        
        markdown_table[method_name] = {"total number of prompts": total_number_of_prompts,
                                       "Avg. temporal distance between prompts": l1_dist_avg,
                                       "total consecutive prompts": cons_p_dist_sum,
                                       "Delta_M": round(delta_P_cummulative_count_AUC_array_mean,3)}

    with _col_2:
        st.subheader(f" - :{ST_COLOR}[Prompt-Info] Table")
        st.text(pd.DataFrame(markdown_table).T.to_markdown())
