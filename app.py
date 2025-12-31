import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Page config
st.set_page_config(page_title="Course Recommender", layout="wide")

# Smart pickle loader (full_data.pkl OR fulldata.pkl)
@st.cache_data
def load_models():
    fulldata_file = 'fulldata.pkl' if os.path.exists('fulldata.pkl') else 'full_data.pkl'
    
    try:
        fulldata = pd.read_pickle(fulldata_file)
        traindata = pd.read_pickle('traindata.pkl')
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
        return fulldata, traindata, tfidf, biases
    except FileNotFoundError as e:
        st.error(f"❌ Missing: {e.filename}")
        st.error("""
**Need these 4 files:**
