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
import pagestyle
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


#connection details from Azure where the pickle files are stored and updated.
connection_string = os.getenv('AZURE_CONNECTION_STRING')
container_name = os.getenv('AZURE_CONTAINER_NAME')

blob_service_client = BlobServiceClient.from_connection_string(connection_string)

@st.cache
def save_to_blob(data, blob_name):
    serialized_data = pickle.dumps(data)
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    blob_client.upload_blob(serialized_data, overwrite=True)
@st.cache
def load_from_blob(blob_name):
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    serialized_data = blob_client.download_blob().readall()
    data = pickle.loads(serialized_data)
    return data


st.set_page_config(page_title="Freecodecamp-YT", page_icon=":bar_chart:", layout="wide")
# Assuming pagestyle.top() sets the top style of the page
pagestyle.top()

# Load image and display it
image = Image.open("logo.jpg")
st.image(image)


#load data that is updated via the daily run on databricks
highest = load_from_blob('top10.pkl')
dfcurr = load_from_blob('video_df.pkl')
totsubs = load_from_blob('totsubs.pkl')
diffsubs = load_from_blob('diffsubs.pkl')
diff =   load_from_blob('diff.pkl')
difflike = load_from_blob('diflikes.pkl')
diffcomment = load_from_blob('difcomment.pkl')
diffview = load_from_blob('difview.pkl')



from math import log, floor
def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P','E','Z']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])



st.header("____Total Engagement____")
st.markdown("The statistics are refreshed daily at 6 AM EST.")
#specify column containers 
cache = st.empty()

#total engagement stats
with cache.container():
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric(label="Total Videos", value=dfcurr.shape[0], delta =dfcurr.shape[0] - dfcurr[diff:].shape[0])
    col2.metric(label="Total Views", value = human_format(dfcurr.viewCount.sum()), 
                delta =(diffview))
    col3.metric(label='Total Subscribers',value = human_format(totsubs), delta = (diffsubs))
    col4.metric(label="Total Likes", value=human_format(dfcurr.likeCount.sum()), delta =(difflike))
    col5.metric(label="Total Comments", value =human_format(dfcurr.commentCount.sum()), delta =(diffcomment))

                
"____"
def fig_config(title, size, size2, xtitle, ytitle):
    fig.update_layout(
        height=500,
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            font=dict(
                family="Arial",
                size=size,
                color='black' # Use "black" for the color
            )
        ),
        xaxis_title=dict(
            text=xtitle,
            font=dict(
                family="Arial",
                size=size2,
                color='black',
            )
        ),
        yaxis_title=dict(
            text=ytitle,
            font=dict(
                family="Arial",
                size=size2,
                color='black',
            )
        ),
    ) 
    fig.update_traces(
        textfont_size=size2,
        textfont_color="White",
        marker= dict(color=px.colors.sequential.Plotly3)
    )

    # Modify the font properties for the second font
    fig.update_xaxes(tickfont=dict(color='black',family="Arial", size=size2))
    fig.update_yaxes(tickfont=dict(color='black',family="Arial", size=size2))

def fig_config_pie(title, size, size2, colors):
    fig.update_layout(
        height=500,
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            font=dict(
                family="Arial",
                size=size,
                color='black'
            )
        )
    )
    fig.update_traces(
        textfont_size=20,
        textfont_color="white",
        marker=colors
    )

#most viewed videos
fig = px.bar(highest, x='viewCount', y='title',color=highest.index,text_auto='.2s')
#fig.update_traces(textinfo='value')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_coloraxes(showscale=False)
fig_config('<b>Top 10 most viewed videos</b>',16,16,"<b>View Count</b>",'<b>Videos</b>')
st.plotly_chart(fig,use_container_width=True)

"____"

#top10 most liked
liked = load_from_blob('liked10.pkl')
fig = px.bar(liked, x='likeCount', y='title',color=liked.index,text_auto='.2s')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_coloraxes(showscale=False)
fig_config('<b>Top 10 most liked videos</b>',16,16,"<b>Like Count</b>",'<b>Video Title</b>')
st.plotly_chart(fig,use_container_width=True)
"____"

#Most viewed certification courses
cert = load_from_blob('cert10.pkl')
fig = px.bar(cert, x='viewCount', y='title',color=cert.index,text_auto='.2s')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_coloraxes(showscale=False)
fig_config('<b>Top 10 most popular certifications</b>',16,16,"<b>View Count</b>",'<b>Video Title</b>')
st.plotly_chart(fig,use_container_width=True)

"____"

#dayoftheweek uploads
day = load_from_blob('day.pkl')
fig = px.bar(day, y='publishedDayName',color=day.index,text_auto='.2ss')
fig.update_coloraxes(showscale=False)
fig.update_layout(showlegend=False)
fig_config('<b>Day with most videos uploaded</b>',16,16,"<b>Day of the Week</b>",'<b>No.of Videos</b>')
st.plotly_chart(fig,use_container_width=True)

"____"
#count of repeating topics
count =  load_from_blob('countvideos.pkl')
fig =px.pie(count, values=count.values, names=count.index, title ='Technology categories that have more than one video posted', width=800, height=800)
fig.update_traces(textinfo='value')
fig_config_pie('<b>Technologies  that have more than one video posted</b>',18,18,dict(colors=px.colors.qualitative.Plotly))
st.plotly_chart(fig,use_container_width=True)

"____"

#word cloud
st.cache(suppress_st_warning=True)
def all():
     allwords = load_from_blob('wordcloud.pkl')
     wordcloud = WordCloud(width=800, height=300, random_state=1, background_color='black',collocations=False).generate(allwords)
     return wordcloud
wordcloud = all()
fig = plt.figure(figsize = (5, 5))
plt.imshow(wordcloud)
plt.axis("off")
st.pyplot(fig)

pagestyle.footer()



