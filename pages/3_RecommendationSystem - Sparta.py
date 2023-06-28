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
import psycopg2

#page settings
st.set_page_config(page_title="Recommendations", page_icon=":mag_right:", layout="wide")
pagestyle.sidebar()

#pickle files paths
from pathlib import Path
root = Path(".")
my_path = root/'pickle files'

#Elephant sql database connections
from dotenv import load_dotenv
load_dotenv(".env")
DATABASE_URL = os.getenv("DATABASE_URL")

def create_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# Insert records into database (feedback_trackersparta)
def insert_period(period, ratings):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedback_trackersparta (period, ratings) VALUES (%s, %s)", (period, ratings))
    conn.commit()
    conn.close()


pagestyle.top()
page_icon = ":mag_right:"
st.header("Freecodecamp-YT video recommender"+""+page_icon)





indices = pickle.load(open(my_path/'indices2.pkl','rb'))
similarity = pickle.load(open(my_path/'sim2.pkl','rb'))
videos = pickle.load(open(my_path/'videos.pkl','rb'))
video_df = pd.DataFrame(pickle.load(open(my_path/'video_df.pkl','rb')))

##settings
ratings =[1,2,3,4,5]
years = [datetime.today().year] 
months = [calendar.month_name[datetime.now().month]]


#predictor function
st.markdown("<h3 style='font-size: 18px;'>Content Based Hybrid Recommendation Engine</h3>", unsafe_allow_html=True)

video_list = videos
selected_video = st.selectbox(
    "Select a video or enter keywords (pls note: **__the first recommendation will be the selected video itself so as to provide the URL link__**):",
    options=video_list, key="1"
)

if st.button('Show video recommendations', key="4"):
    st.write("Recommended videos:")
    recommended = pagestyle.recommender_pipeline(selected_video, indices, similarity, video_df)
    recommended = recommended.hide(axis="index")  # hide dataframe index
    st.write(recommended.to_html(), unsafe_allow_html=True)
  

"---"


with st.expander("Feedback", expanded=True):
    st.markdown("<h3 style='font-size: 16px;'>Recommendation Feedback</h3>", unsafe_allow_html=True)
    st.write("Based on your interaction with the recommender, please provide your ratings between 1-5, where rating 1 is where you believe overall only 1 rating was useful/relevant and 5 as in all 5 ratings where useful and so on. Kindly ignore the first recommendation as that is your selection")
    col1, col2 = st.columns(2)
    selected_month = col1.selectbox("Select Month:", months, key="month")
    selected_year = col2.selectbox("Select Year:", years, key="year")
    selected_rating = st.selectbox("Select a rating between 1-5, with 1 being low and 5 being the highest rating:", options=ratings, key='ratings')

    if st.button("Submit", key="3"):
        period = str(selected_month) + "_" + str(selected_year)
        insert_period(period, selected_rating)
        st.success("Thank you! Head on over to Live Feedback Tracker.")


"---"



pagestyle.footer()
