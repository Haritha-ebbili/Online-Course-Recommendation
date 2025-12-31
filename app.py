import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Course Recommender", layout="wide")

# Load data and models
@st.cache_data
def load_all():
    df = pd.read_pickle('full_data.pkl')
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    train_df = pd.read_pickle('train_data.pkl')
    biases = pickle.load(open('biases.pkl', 'rb'))
    return df, tfidf, train_df, biases

df, tfidf, train_df, biases = load_all()
global_mean = biases['global_mean']
user_bias = biases['user_bias']
item_bias = biases['item_bias']

# Simple recommendation function
def get_user_recommendations(user_id=15796, n=5):
    # User courses already taken
    user_courses = set(df[df['userid'] == user_id]['courseid'].unique())
    
    # All other courses
    candidates = df[~df['courseid'].isin(user_courses)].copy()
    
    # Calculate simple score
    candidates['score'] = candidates['courseid'].apply(
        lambda x: global_mean + user_bias.get(user_id, 0) + item_bias.get(x, 0)
    )
    
    # Top 5
    top = candidates.nlargest(n, 'score')
    
    return top[['courseid', 'score', 'coursename', 'instructor', 'rating']]

# UI
st.title("🎓 Course Recommendations")

# User input
user_id = st.number_input("User ID", value=15796, min_value=1, max_value=49999)

# SHOW RECOMMENDATIONS IMMEDIATELY
st.markdown(f"## Top Recommendations for User {user_id}")

recommendations = get_user_recommendations(user_id)

# EXACT TABLE
st.markdown("**course_id | recommendation_score | course_name | instructor | rating**")

for _, row in recommendations.iterrows():
    st.markdown(f"""
    **{int(row['courseid'])}** | **{row['score']:.3f}** | **{row['coursename']}** | **{row['instructor']}** | **{row['rating']:.1f}**
    """)

st.success("✅ Working!")
