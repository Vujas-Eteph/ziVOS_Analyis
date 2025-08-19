# - IMPORTS ---
import streamlit as st


# - FUNCTION ---
def streamlit_custom_page_configuration(
    title="Find_title",
    emoji=""
):
    custom_css = """
<style>
    /* Apply monospace font to the whole page, including headers and sidebars */
    html, body, [class*="st-"], .stTextInput>div>div>input, .stSelectbox>div>div>select {
        font-family: 'Courier New', monospace;
    }
    /* Specifically target headings, sidebar, and markdown text for monospace font */
    h1, h2, h3, h4, h5, h6, .stMarkdown, .css-1d391kg, .css-hi6a2p {
        font-family: 'Courier New', monospace !important;
    }
    /* Apply monospace font to sidebar elements */
    .stSidebar > div, .stSidebar .css-1d391kg, .stSidebar .css-hi6a2p {
        font-family: 'Courier New', monospace !important;
    }
</style>
"""
    st.set_page_config(layout="wide", page_title=title, page_icon=emoji)
    st.markdown(custom_css, unsafe_allow_html=True)
