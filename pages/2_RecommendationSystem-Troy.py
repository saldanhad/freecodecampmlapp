

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import requests
import calendar
from datetime import datetime
import pagestyle

#page settings
st.set_page_config(page_title="Recommendations", page_icon=":mag_right:", layout="wide")
#adjust sidebar settings
pagestyle.sidebar()

from pathlib import Path
root = Path(".")
my_path = root/'pickle files'


#no sql database connections
from deta import Deta
from dotenv import load_dotenv
load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")

#Initialize with a project key
deta = Deta(DETA_KEY)

db = deta.Base("Feedback_TrackerTroy")


#remove space at the top
pagestyle.top()

page_icon = ":mag_right:"
st.header("Freecodecamp-YT video recommender"+""+page_icon)


#load pickled files
indices = pickle.load(open(my_path/'indices.pkl','rb'))
similarity = pickle.load(open(my_path/'sim.pkl','rb'))
videos = pickle.load(open(my_path/'videos.pkl','rb'))
video_df = pd.DataFrame(pickle.load(open(my_path/'video_df.pkl','rb')))


##settings
ratings =[1,2,3,4,5]
years = [datetime.today().year] 
months = [calendar.month_name[datetime.now().month]]


#predictor function


with st.expander("Hybrid Recommendation engine", expanded=True):
    video_list = videos
    selected_video = st.selectbox(
        "Select a video or enter keywords (pls note: **__the first recommendation will be the selected video itself so as to provide the url link__**):",
        options=video_list,key="1")

    if st.button('Show video recommendations',key="4"):
        st.write("Recommended videos :")
        recommended = pagestyle.recommender(selected_video,indices, similarity,video_df)
        recommended = recommended.hide(axis="index") #hide dataframe index
        st.write(recommended.to_html(), unsafe_allow_html=True)  

"---"



pagestyle.stratings(db,months, years,ratings,"Thank you!, head on over to Sparta for more recommendations")
        
"---"

def get_all_periods():
    items = db.fetch_all_periods()
    periods = [item["period"] for item in items]
    return periods


#define footer for this page
pagestyle.footer()
