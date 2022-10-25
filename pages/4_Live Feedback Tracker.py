import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.express as px
from deta import Deta
from dotenv import load_dotenv
import plotly.graph_objects as go
import pagestyle



st.set_page_config(page_title="Live Feedback", page_icon=":chart_with_upwards_trend:", layout="wide")
pagestyle.sidebar()

pagestyle.top()
st.header("Live Feedback"+":chart_with_upwards_trend:")


page_icon = ":chart_with_upwards_trend:"

st.markdown("**__An important way of evaluating recommendation systems is getting the real-time performance feedback of the users, the below plot provides a real time view of the feedback received from the users of the recommender__**")

st.markdown("**___Troy System___**")
#no sql database connections
load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")
#Initialize with a project key
deta = Deta(DETA_KEY)

db = deta.Base("Feedback_TrackerTroy")

#Plot ratings view in Recommendation Feedback
def feedback(db): 
    res = db.fetch().items #fetch all items from database as list
    df = pd.DataFrame(res)
    fig=(px.histogram(df, x='period', color='ratings'))
    return fig


def accuracy(db):
    res = db.fetch().items #fetch all items from database as list
    df = pd.DataFrame(res)
    accuracy = round(df.ratings.sum()/(df.shape[0]*5) * 100)
    return accuracy

figsparta = st.empty()
figtroy = feedback(db)
st.plotly_chart(figtroy)

#python table using plotly

res = db.fetch().items #fetch all items from database as list
df = pd.DataFrame(res)

fig2troy = go.Figure(data=[go.Table(columnwidth = [300,300],header=dict(values=[['<b>Relevant</b><br>'], ['<b>Recommendations</b><br>']],
line_color='darkslategray',
    align=['center','center'],
    font=dict(color='black', size=18),
    height=40
  ),
                 cells=dict(values=[df.ratings.sum(), df[df.columns[0]].count()*5], fill_color='lavender',
    line_color='darkslategray',
    fill=dict(color=['paleturquoise', 'white']),
    align=['center', 'center'],
    font_size=16,
    height=30))
    ])

st.plotly_chart(fig2troy)
acctroy = st.empty()
acctroy = accuracy(db)
st.write("___Accuracy=___",acctroy)

"____"

st.write("**___Sparta System___**")

db2 = deta.Base("Feedback_TrackerSparta")

figsparta = st.empty()
figsparta = feedback(db2)
st.plotly_chart(figsparta)

#python table using plotly

res = db2.fetch().items #fetch all items from database as list
df2 = pd.DataFrame(res)

fig2sparta = go.Figure(data=[go.Table(columnwidth = [300,300],header=dict(values=[['<b>Relevant</b><br>'], ['<b>Recommendations</b><br>']],
line_color='darkslategray',
    align=['center','center'],
    font=dict(color='black', size=18),
    height=40
  ),
                 cells=dict(values=[df2.ratings.sum(), df2[df2.columns[0]].count()*5], fill_color='lavender',
    line_color='darkslategray',
    fill=dict(color=['paleturquoise', 'white']),
    align=['center', 'center'],
    font_size=16,
    height=30))
    ])

st.plotly_chart(fig2sparta)
accsparta = st.empty()
accsparta = accuracy(db2)
st.write("___Accuracy=___",accsparta)


#hide main menu 
pagestyle.footer()