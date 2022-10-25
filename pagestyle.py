import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd


#remove space at top of page
st.cache(suppress_st_warning=True)
def top():
    topstyle = """
        <style>
               .css-18e3th9 {
                    padding-top: 1rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """
    st.markdown(topstyle, unsafe_allow_html=True)


st.cache(suppress_st_warning=True)
def footer():
    #hide main menu 
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            a:link , a:visited{
            color: blue;
            background-color: transparent;
            }

            a:hover,  a:active {
            color: red;
            background-color: transparent;
            }

            .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: white;
            color: black;
            text-align: center;
            }
            </style>
            <div class="footer">
            <p>Built by <a text-align: center;' href="https://github.com/saldanhad" target="blank" style="text-decoration:none;">Deepaks Saldanha</a></p>
            </div>
            """

    st.markdown(hide_menu_style, unsafe_allow_html=True)

st.cache(suppress_st_warning=True)
def sidebar():
    sidebar = """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 243px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 245px;
            margin-left: -245px;
        }
        
        """
    st.markdown(sidebar,unsafe_allow_html=True)

    
#recommender pipeline 
st.cache(suppress_st_warning=True)
def recommender(selected_video,indices, similarity, video_df):
    index = indices[selected_video]
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[0:6]
    vid_indices = [i[0] for i in sim_scores]
    df=video_df['title'].iloc[vid_indices]
    df=df.to_frame()
    df= pd.merge(df, video_df[['likeCount']], left_index=True, right_index=True) #include count of likes
    df0 = df[:1] # to show row for the selected video itself
    df = df[1:6].sort_values(by=['likeCount'], ascending=False)
    df = pd.concat([df0,df])
    df['url']= video_df['url'].iloc[vid_indices]
    def make_clickable(output):
        return '<a target="_blank" href="{}">{}</a>'.format(output,output)
    df = df.style.format({'url':make_clickable}) #make url clickable
    return df


st.cache(suppress_st_warning=True)
def stratings(db,months,years,ratings,msg):
    #insert_period = db.put({"period":period,"ratings":ratings})
    with st.expander("Recommendation Feedback", expanded=True):
        st.write("Based on your interaction with the recommender, please provide your ratings between 1-5, where rating 1 is where you believe overall only 1 rating was useful/relevant and 5 as in all 5 ratings where useful and so on. Kindly ignore the first recommendation as that is your selection")
        col1, col2= st.columns(2)
        col1.selectbox("Select Month:", months, key="month")
        col2.selectbox("Select Year:", years, key="year")
        st.selectbox("Select a rating between 1-5, with 1 being low and 5 being the highest rating:",options=ratings,key='ratings')

        if st.button("Submit", key="3"):
            #initialize session state for ratings, month, year
            if 'ratings' not in st.session_state:
                st.session_state['ratings'] = 0
            if 'year' not in st.session_state:
                st.session_state['year'] =0
            if 'month' not in st.session_state:
                st.session_state['month'] =0
            period = str(st.session_state["month"])+"_"+str(st.session_state["year"])
            ratings = (st.session_state['ratings'])
            def insert_period(period,ratings):
                return db.put({"period":period,"ratings":ratings})
            insert_period(period,ratings)
            st.success(msg)
        

if __name__ == '__main__':
    top()
    sidebar()
    footer()
    recommender()
    stratings()