import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_pickle('full_data.pkl')
    return df

@st.cache_data
def load_models():
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    train_df = pd.read_pickle('train_data.pkl')
    biases = pickle.load(open('biases.pkl', 'rb'))
    return tfidf, train_df, biases

# Load
df = load_data()
tfidf, train_df, biases = load_models()
global_mean = biases['global_mean']
user_bias = biases['user_bias']
item_bias = biases['item_bias']

# Find display columns dynamically
user_col = next(col for col in df.columns if 'user' in col.lower())
course_col = next(col for col in df.columns if 'course' in col.lower() and 'id' in col.lower())
name_col = next(col for col in df.columns if 'name' in col.lower() and 'course' in col.lower())
instructor_col = next((col for col in df.columns if 'instructor' in col.lower()), 'instructor')
price_col = next((col for col in df.columns if 'price' in col.lower()), 'courseprice')

def content_predict(course_id):
    course_info = df[df[course_col] == course_id].iloc[0]
    course_text = f"{course_info[instructor_col]} {course_info[name_col]}"
    course_features = tfidf.transform([course_text])
    
    all_text = df[name_col].astype(str) + ' ' + df[instructor_col].astype(str)
    all_features = tfidf.transform(all_text)
    similarities = cosine_similarity(course_features, all_features).flatten()
    
    weights = similarities[:len(train_df)]
    ratings = train_df['rating'].values
    valid = weights > 0
    return np.mean(ratings[valid] * weights[valid]) if valid.sum() > 0 else global_mean

def cf_predict(user_id, course_id):
    bu = user_bias.get(user_id, 0)
    bi = item_bias.get(course_id, 0)
    return global_mean + bu + bi

def hybrid_score(user_id, course_id):
    return 0.6 * cf_predict(user_id, course_id) + 0.4 * content_predict(course_id)

def get_recommendations(user_id, n=5, max_price=500, min_rating=3.0):
    """EXACTLY matches your output format for User 15796"""
    # Get user's unseen courses
    user_courses = set(df[df[user_col] == user_id][course_col].unique())
    candidates = df[~df[course_col].isin(user_courses)].copy()
    
    # Filter
    candidates = candidates[candidates['rating'] >= min_rating]
    candidates = candidates[candidates[price_col] <= max_price]
    
    # Calculate scores
    candidates['recommendation_score'] = candidates[course_col].apply(lambda x: hybrid_score(user_id, x))
    
    # Sort and get top 5 EXACTLY like your output
    top_recs = candidates.nlargest(5, 'recommendation_score')
    
    # Select EXACT columns in EXACT order
    result = top_recs[[course_col, 'recommendation_score', name_col, instructor_col, 'rating']].copy()
    result.columns = ['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']
    
    return result

# === UI ===
st.markdown("# 🎓 Top Recommendations for User 15796")
st.markdown("---")

st.sidebar.header("Controls")
user_id = st.sidebar.number_input("User ID", 1, 49999, 15796, key="user_input")
show_n = st.sidebar.slider("Show Top", 1, 5, 5)

# Get recommendations
recommendations = get_recommendations(user_id, show_n)

# EXACT TABLE FORMAT like your output
if not recommendations.empty:
    st.markdown("**course_id | recommendation_score | course_name | instructor | rating**")
    
    for idx, row in recommendations.iterrows():
        course_name_short = (row['course_name'][:25] + "...") if len(row['course_name']) > 25 else row['course_name']
        instructor_short = (row['instructor'][:12] + "...") if len(row['instructor']) > 12 else row['instructor']
        
        st.markdown(f"""
        **{int(row['course_id'])}** | **{row['recommendation_score']:.3f}** | 
        **{course_name_short}** | **{instructor_short}** | **{row['rating']:.1f}**
        """)
else:
    st.warning("No recommendations found!")

st.markdown("---")
st.caption("✅ Exact format match for User 15796 recommendations [file:1][file:2]")
