import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import pagestyle

#pickle files paths
from pathlib import Path
root = Path(".")
my_path = root/'pickle files'


#st.set_page_config(page_title="Search Engine", page_icon=":chart_with_upwards_trend:", layout="wide")
pagestyle.sidebar()

st.title("Search Videos by keyword")
keyword = pickle.load(open(my_path/'keyword.pkl', 'rb'))


with st.expander("Search All videos on the freecodecamp channel", expanded=True):
    text_input = st.text_input("Enter a keyword, please use lower case","ex: python, java, aws, html")

    if st.button('Show videos',key="4"):
        st.write("Recommended videos :")
        videos = keyword.loc[(keyword[['key1','key2','key3']] == text_input).any(axis=1)]
        videos = videos[['title','url']]
        def make_clickable(val):
            return '<a target="_blank" href="{}">{}</a>'.format(val,val)
        videos = videos.style.format({'url':make_clickable})
        videos = videos.hide(axis='index')
        st.write(videos.to_html(), unsafe_allow_html=True)


pagestyle.footer()