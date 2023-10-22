import os
import streamlit as st
import numpy as np
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
import pagestyle

st.set_page_config(page_title="Live Feedback", page_icon=":chart_with_upwards_trend:", layout="wide")
pagestyle.sidebar()

pagestyle.top()
st.header("Live Feedback" + ":chart_with_upwards_trend:")

page_icon = ":chart_with_upwards_trend:"

st.markdown("**__An important way of evaluating recommendation systems is getting the real-time performance feedback of the users, the below plot provides a real-time view of the feedback received from the users of the recommender__**")

st.markdown("**___Troy System___**")

# ElephantSQL PostgreSQL database connection
DATABASE_URL = os.getenv("DATABASE_URL")

def create_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

#retrieve only required columns from feedback_trackertroy
def feedback(table_name):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT period, ratings FROM {table_name}")
    res = cursor.fetchall()
    df = pd.DataFrame(res, columns=["period", "ratings"])
    fig = px.histogram(df, x='period', color='ratings')
    conn.close()
    return fig

def accuracy(table_name):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT period,ratings FROM {table_name}")
    res = cursor.fetchall()
    df = pd.DataFrame(res, columns=["period", "ratings"])
    accuracy = round(df["ratings"].sum() / (df.shape[0] * 5) * 100)
    conn.close()
    return accuracy

#performance for troy model
figtroy = feedback("feedback_trackertroy")
st.plotly_chart(figtroy)

acc_troy = accuracy("feedback_trackertroy")
st.markdown(f"<h3 style='font-size: 24px;'>Accuracy= {acc_troy}</h3>", unsafe_allow_html=True)

"---"

st.markdown("**___Sparta System___**")

#performance for sparta model
figsparta = feedback("feedback_trackersparta")
st.plotly_chart(figsparta)

acc_sparta = accuracy("feedback_trackersparta")
st.markdown(f"<h3 style='font-size: 24px;'>Accuracy= {acc_sparta}</h3>", unsafe_allow_html=True)

"---"

pagestyle.footer()
