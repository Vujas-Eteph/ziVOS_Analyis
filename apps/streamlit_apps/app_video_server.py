"""
Run a streamlit page to visualize the videos (results).

Quick run:
- streamlit run ./streaming_apps/app_video_server.py

TODO: Add more result visualization??

by Stéphane Vujasinovic
"""

# - IMPORTS ---
import streamlit as st
import os

# - CONSTANTS ---
DATASET_OPTIONS = ["d17-val", "lvos-val"]
METHODS_OPTIONS = ["XMem"]

# - FUNCTIONS ---


# - MAIN ---
if __name__ == "__main__":
    st.header(":rainbow[Entropy Results]", divider="rainbow")
    with st.sidebar:
        dataset_name = st.selectbox(('Dataset Option'),
                                    options=DATASET_OPTIONS,)
        method_name = st.selectbox(('Method Option'),
                                   options=METHODS_OPTIONS)

    # Make sure to generate the correspdong videos first, with app_visualize_entropy.py or app_opencv.py
    PATH_TO_VIDEOS_ASSETS = f"../../experimental_assets/videos/{dataset_name}/{method_name}/Entropy"

    sequences = os.listdir(PATH_TO_VIDEOS_ASSETS)
    for sequence in sequences:
        st.subheader(f"Sequence: :rainbow[{sequence.rstrip('.webm')}]",
                    divider='rainbow')
        video_path = os.path.join(PATH_TO_VIDEOS_ASSETS, sequence)
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)
