import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import pagestyle
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

connection_string = os.getenv('AZURE_CONNECTION_STRING')
container_name = os.getenv('AZURE_CONTAINER_NAME')

blob_service_client = BlobServiceClient.from_connection_string(connection_string)

def save_to_blob(data, blob_name):
    serialized_data = pickle.dumps(data)
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    blob_client.upload_blob(serialized_data, overwrite=True)

def load_from_blob(blob_name):
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    serialized_data = blob_client.download_blob().readall()
    data = pickle.loads(serialized_data)
    return data


#st.set_page_config(page_title="Search Engine", page_icon=":chart_with_upwards_trend:", layout="wide")
pagestyle.sidebar()

st.title("Search Videos by keyword")
keyword = load_from_blob('keyword.pkl')


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
