import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import requests
import calendar
from datetime import datetime
import plotly.express as px
#call the local directory
from pathlib import Path
import pickle
import pagestyle
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud


root = Path(".")
my_path = root/'pickle files'




#https://www.webfx.com/tools/emoji-cheat-sheet/

st.set_page_config(page_title="Freecodecamp-YT", page_icon=":bar_chart:", layout="wide")
pagestyle.top()

image = Image.open("logo.jpg")
st.image(image)
page_icon = ":bar_chart:"
st.header("Near Real-time Freecodecamp-YT Channel Analytics Dashboard & Recommender WebApp"+""+page_icon)


st.markdown("__Freecodecamp is a non-profit organization helping millions to learn to code for free. This WebApp is built as an additional resource that provides users with overall statistics and recommendations of videos on the freecodecamp youtube channel__")
st.markdown("__This is a multipage app, please use the side bar to navigate to other pages. This website is best viewed on a laptop/desktop__")



pagestyle.sidebar()



#st.dataframe(df)
highest = pickle.load(open(my_path/'top10.pkl','rb')) #top20 most viewed videos

dfcurr = pickle.load(open(my_path/'dfcurr.pkl','rb'))
diff = pickle.load(open(my_path/'dfdiff.pkl','rb'))
totsubs = pickle.load(open(my_path/'totsubs.pkl','rb'))
diffsubs = pickle.load(open(my_path/'diffsubs.pkl','rb'))
subsold = pickle.load(open(my_path/'subsold.pkl','rb'))
subsnew = pickle.load(open(my_path/'subsnew.pkl','rb'))

from math import log, floor
@st.cache(suppress_st_warning=True)
def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P','E','Z']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])


st.header("____Total Engagement____")

#specify column containers 
cache = st.empty()
with cache.container():
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric(label="Total Videos", value=dfcurr.shape[0], delta =dfcurr.shape[0] - dfcurr[diff:].shape[0])
    col2.metric(label="Total Views", value = human_format(dfcurr.viewCount.sum()), 
                delta =human_format(dfcurr.viewCount.sum() - dfcurr.viewCount[diff:].sum()))
    col3.metric(label='Total Subscribers',value = human_format(totsubs), delta = human_format(diffsubs))
    col4.metric(label="Total Likes", value=human_format(dfcurr.likeCount.sum()), delta =human_format(dfcurr.likeCount.sum() - dfcurr.likeCount[diff:].sum()))
    col5.metric(label="Total Comments", value =human_format(dfcurr.commentCount.sum()), delta =human_format(dfcurr.commentCount.sum() - dfcurr.commentCount[diff:].sum()))


"___"

@st.cache(suppress_st_warning=True)
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
"___"

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
fig = px.bar(day, y='publishedDayName',color=day.index,text_auto='.2s')
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

import os
from googleapiclient.discovery import build
import pandas as pd 
from dateutil import parser
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#nltk packages
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('omw-1.4')
import string
from nltk.stem import WordNetLemmatizer

from dotenv import load_dotenv
load_dotenv(".env")
API_KEY = os.getenv("API_KEY")

channel_ids = ['UC8butISFwT-Wl7EV0hUK0BQ']

api_service_name = "youtube"
api_version = "v3"

# Get credentials and create an API client
youtube = build(api_service_name, api_version, developerKey=API_KEY)

from pathlib import Path
root = Path(".")
my_path = root/'pickle files'

dfold = pickle.load(open(my_path/'video_df.pkl','rb'))


def get_video_ids(youtube, playlist_id):
    
    video_ids = []
    
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults = 50
    )
    response = request.execute()
    
    for item in response['items']:
        video_ids.append(item['contentDetails']['videoId'])
        
    next_page_token = response.get('nextPageToken')
    while next_page_token is not None:
        request = youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId = playlist_id,
                    maxResults = 50,
                    pageToken = next_page_token)
        response = request.execute()

        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page_token = response.get('nextPageToken')
        
    return video_ids

playlist_id = 'UU8butISFwT-Wl7EV0hUK0BQ'
video_ids = get_video_ids(youtube, playlist_id)



def get_video_details(youtube, video_ids):

    all_video_info = [] # instantiate empty list
    
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute() 

        for video in response['items']:
            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                             'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],
                             'contentDetails': ['duration', 'definition', 'caption']
                            }
            video_info = {} #instantiate empty dictionary
            video_info['video_id'] = video['id']

            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    #implement try try except block to retrieve all videos including those without tags
                    try:  
                        video_info[v] = video[k][v]
                    except:
                        video_info[v] = None

            all_video_info.append(video_info) #append results from video_info dict to all_video_info list
    
    return pd.DataFrame(all_video_info) #covert list to dataframe

#channel resource contains information about a youtube channel
#use the list method to gather channel information by specifying the channel id 


def get_channel_stats(youtube, channel_ids):
    
    all_data = [] #initialize empty list
    
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=','.join(channel_ids)
    )
    response = request.execute()

    # loop through items
    for item in response['items']:
        data = {'channelName': item['snippet']['title'],
                'subscribers': item['statistics']['subscriberCount'],
                'hiddensubscriber':item['statistics']['hiddenSubscriberCount'],
                'views': item['statistics']['viewCount'],
                'playlistId': item['contentDetails']['relatedPlaylists']['uploads']
        }
        
        all_data.append(data)
        
    return(pd.DataFrame(all_data))

channel_stats = get_channel_stats(youtube, channel_ids)
channel_stats['subscribers'] = channel_stats['subscribers'].apply(pd.to_numeric, errors = 'coerce')

subsold = pickle.load(open(my_path/'subsold.pkl','rb'))
subsnew = channel_stats.copy()


#update diff of subscribers only when there is a change in values.
if subsold.subscribers.any() != subsnew.subscribers.any():
    diffsubs = subsnew.subscribers[0] - subsold.subscribers[0]
    with open(my_path/'diffsubs.pkl','wb') as f:
        pickle.dump(diffsubs,f)
    with open(my_path/'subsold.pkl','wb') as f:
        pickle.dump(subsnew,f)

totsubs = channel_stats.subscribers

with open(my_path/'totsubs.pkl','wb') as f:
    pickle.dump(totsubs,f)

#video_df handle datatypes and create published day name
video_df = get_video_details(youtube, video_ids)
#define url
video_df['url'] = 'https://www.youtube.com/watch?v='+video_df.video_id
#handle datatypes for columns
numeric_cols = ['viewCount', 'likeCount', 'favouriteCount', 'commentCount']
video_df[numeric_cols] = video_df[numeric_cols].apply(pd.to_numeric, errors = 'coerce', axis = 1)
video_df['publishedAt'] = video_df['publishedAt'].astype(str)
video_df['publishedAt'] = video_df['publishedAt'].apply(lambda x:parser.parse(x))
video_df['publishedDayName'] = video_df['publishedAt'].apply(lambda x:x.strftime("%A"))


#capture new incoming videos and their statistics
dfcurr = video_df.copy()

diff = dfcurr.shape[0] - dfold.shape[0]

with open(my_path/'dfcurr.pkl','wb') as f:
    pickle.dump(dfcurr,f)


#whatever is the number of new videos uploaded that is tracked. Similar to what we have done for diff for subscribers.
#pickle the diff calculated above here


def check_data():
    if dfold.shape[0] != dfcurr.shape[0]:

        with open(my_path/'dfdiff.pkl','wb') as f:
            pickle.dump(diff,f)
        with open(my_path/'video_df.pkl','wb') as f:
            pickle.dump(video_df,f)
        
        #generate clean_title & clean_description for incoming data
        VERB_CODES = {'VB','VBD','VBG','VBN','VBP','VBZ'}
        stop_words = set(stopwords.words('english'))
        lemma = WordNetLemmatizer()

        def preprocess_text(text):
            text = text.lower()
            temp = [] #initialize dictionary
            words = nltk.word_tokenize(text)
            tags = nltk.pos_tag(words)
            for i, word in enumerate(words):
                if tags[i][1] in VERB_CODES:
                    lemmatized = lemma.lemmatize(word,'v')
                else:
                    lemmatized = lemma.lemmatize(word)
                if lemmatized not in stop_words and lemmatized.isalpha():
                    temp.append(lemmatized)
            final = ' '.join(temp)
            return final

        video_df['clean_description'] = video_df['description'].apply(preprocess_text)
        video_df['clean_title'] = video_df['title'].apply(preprocess_text)
        
        #update all related pickle files

        #top10 viewcount
        top10 = video_df[['title','video_id','viewCount']]
        top10 = top10.sort_values(by='viewCount',ascending=False).head(10)
        with open(my_path/'top10.pkl','wb') as f:
            pickle.dump(top10,f)
        
        #top10 mostliked
        liked10 = video_df[['title','video_id','likeCount']]
        liked10 = liked10.sort_values(by='likeCount',ascending=False).head(20)
        with open(my_path/'like10.pkl','wb') as f:
            pickle.dump(liked10,f)

        #update keyword search
        video_df['key1']=video_df['clean_title'].str.split().str[0]
        video_df['key2']=video_df['clean_title'].str.split().str[1]
        video_df['key3']=video_df['clean_title'].str.split().str[2]
        keyword= video_df[['title','url','key1','key2','key3']]

        with open(my_path/'keyword.pkl', 'wb') as f:
            pickle.dump(keyword,f)

        #most viewed certification courses
        cloud = video_df.loc[(video_df[['key1','key2','key3']].isin(['aws','certification','associate','practitioner','azure','google'])).any(axis=1)]
        c = cloud.sort_values(by='viewCount',ascending=False).head(10)
        with open(my_path/'cert10.pkl','wb') as f:
            pickle.dump(c,f)
        
        #wordcloud
        all_words = list([a for b in video_df['title'].to_list() for a in b])
        all_words_str = ''.join(all_words)
        with open(my_path/'wordcloud.pkl','wb') as f:
            pickle.dump(all_words_str,f)
        
        #publishedDayName
        day = pd.DataFrame(video_df['publishedDayName'].value_counts())
        weekdays =['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        day = day.reindex(weekdays)
        ax = day.reset_index().plot.bar(x='index',y='publishedDayName',rot=90)
        with open(my_path/'day.pkl','wb') as f:
            pickle.dump(day,f)
        
        #videocounts of repeated topics
        video_df['clean_title'].value_counts()
        count = video_df['clean_title'].value_counts()
        count = count.loc[count > 1]
        with open(my_path/'countvideos.pkl','wb') as f:
            pickle.dump(count,f)

        #Update pickle files for Troy
        tfv = TfidfVectorizer()
        
        tfv_matrix = tfv.fit_transform(video_df['clean_description'])
        similarity = cosine_similarity(tfv_matrix, tfv_matrix)
        indices = pd.Series(video_df.index, index=video_df['title'])
        videos = video_df['title']

        import pickle
        with open(my_path/'sim.pkl','wb') as f:
            pickle.dump(similarity,f)

        with open(my_path/'indices.pkl','wb') as f:
            pickle.dump(indices,f)

        with open(my_path/'videos.pkl','wb') as f:
            pickle.dump(videos,f)

        with open(my_path/'video_df.pkl','wb') as f:
            pickle.dump(video_df,f)

        #update pickle files for Sparta
        tfv2 = TfidfVectorizer(min_df=2, max_features=None,strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3),stop_words='english')

        tfv2_matrix = tfv.fit_transform(video_df['title'])
        similarity2 = cosine_similarity(tfv2_matrix, tfv2_matrix)

        indices2 = pd.Series(video_df.index, index=video_df['title']).drop_duplicates()

        with open(my_path/'sim2.pkl','wb') as f:
            pickle.dump(similarity2,f)

        with open(my_path/'indices2.pkl','wb') as f:
            pickle.dump(indices2,f)

check_data()
