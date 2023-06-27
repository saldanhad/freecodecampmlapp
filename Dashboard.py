import os
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import log, floor




st.set_page_config(page_title="Freecodecamp-YT", page_icon=":bar_chart:", layout="wide")
# Assuming pagestyle.top() sets the top style of the page
pagestyle.top()

# Load image and display it
image = Image.open("logo.jpg")
st.image(image)


@st.cache
def load_pickle_file(file_path):
    return pickle.load(open(file_path, 'rb'))

# Load pickle files asynchronously
my_path = Path(".")/'pickle_files'
with ThreadPoolExecutor() as executor:
    highest = executor.submit(load_pickle_file, my_path/'top10.pkl')
    dfcurr = executor.submit(load_pickle_file, my_path/'video_df.pkl')
    totsubs = executor.submit(load_pickle_file, my_path/'totsubs.pkl')
    diffsubs = executor.submit(load_pickle_file, my_path/'diffsubs.pkl')
    diff = executor.submit(load_pickle_file, my_path/'diff.pkl')
    diflike = executor.submit(load_pickle_file, my_path/'diflikes.pkl')
    difcomment = executor.submit(load_pickle_file, my_path/'difcomment.pkl')
    difview = executor.submit(load_pickle_file, my_path/'difview.pkl')


from math import log, floor
def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P','E','Z']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])



st.header("____Total Engagement____")
st.markdown("stats updated for every new video added/removed")
#specify column containers 
cache = st.empty()

with cache.container():
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric(label="Total Videos", value=dfcurr.shape[0], delta =dfcurr.shape[0] - dfcurr[diff:].shape[0])
    col2.metric(label="Total Views", value = human_format(dfcurr.viewCount.sum()), 
                delta =(difview))
    col3.metric(label='Total Subscribers',value = human_format(totsubs), delta = (diffsubs))
    col4.metric(label="Total Likes", value=human_format(dfcurr.likeCount.sum()), delta =(diflike))
    col5.metric(label="Total Comments", value =human_format(dfcurr.commentCount.sum()), delta =(difcomment))

                
"____"
def fig_config(title,size,size2,color,xtitle,ytitle):
    fig.update_layout(
    height=500,
    title=dict(
        text=title,
        x=0.5,
        y=0.95,
        font=dict(
            family="Arial",
            size=size,
            color=color
        )
    ),
    xaxis_title=xtitle,
    yaxis_title=ytitle,
    font=dict(
        size=size2,
    )
)
                
                
#most viewed videos
fig = px.bar(highest, x='viewCount', y='title',color=highest.index,text_auto='.2s')
#fig.update_traces(textinfo='value')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_coloraxes(showscale=False)
fig_config('<b>Top 10 most viewed videos</b>',16,14,'#000000',"<b>View Count</b>",'<b>Videos</b>')
st.plotly_chart(fig,use_container_width=True)

"____"

#top10 most liked
liked = pickle.load(open(my_path/'liked10.pkl','rb'))
fig = px.bar(liked, x='likeCount', y='title',color=liked.index,text_auto='.2s')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_coloraxes(showscale=False)
fig_config('<b>Top 10 most liked videos</b>',16,14,'#000000',"<b>Count</b>",'<b>Video Title</b>')
st.plotly_chart(fig,use_container_width=True)
"____"

#Most viewed certification courses
cert = pickle.load(open(my_path/'cert10.pkl','rb'))
fig = px.bar(cert, x='viewCount', y='title',color=cert.index,text_auto='.2s')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_coloraxes(showscale=False)
fig_config('<b>Top 10 most popular certifications</b>',16,14,'#000000',"<b>View Count</b>",'<b>Video Title</b>')
st.plotly_chart(fig,use_container_width=True)

"____"

#dayoftheweek uploads
day = pickle.load(open(my_path/'day.pkl','rb'))
fig = px.bar(day, y='publishedDayName',color=day.index,text_auto='.2ss')
fig.update_coloraxes(showscale=False)
fig.update_layout(showlegend=False)
fig_config('<b>Day with most videos uploaded</b>',16,14,'#000000',"<b>Day of the Week</b>",'<b>No.of Videos</b>')
st.plotly_chart(fig,use_container_width=True)

"____"
#count of repeating topics
count = pickle.load(open(my_path/'countvideos.pkl','rb'))
fig =px.pie(count, values=count.values, names=count.index, title ='Technology categories that have more than one video posted', width=800, height=800)
fig.update_traces(textinfo='value')
fig_config('<b>Technologies  that have more than one video posted</b>',18,14,'#000000','<b>Technology categories that have more than one video posted</b>','<b>Technology categories that have more than one video posted</b>')
st.plotly_chart(fig,use_container_width=True)

"____"

#word cloud
st.cache(suppress_st_warning=True)
def all():
     allwords = pickle.load(open(my_path/'wordcloud.pkl','rb'))
     wordcloud = WordCloud(width=800, height=300, random_state=1, background_color='black',collocations=False).generate(allwords)
     return wordcloud
wordcloud = all()
fig = plt.figure(figsize = (5, 5))
plt.imshow(wordcloud)
plt.axis("off")
st.pyplot(fig)

pagestyle.footer()



#the new videos uploaded are tracked. Similar to what we have done for diff for subscribers.
#pickle the diff calculated above here

import realtimedata
#realtimedata.api_call()
realtimedata.update_recommendations()

