import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Online Course Recommender", layout="wide")

# Load dataset and models
@st.cache_data
def load_data():
    df = pd.read_excel('online_course_recommendation.xlsx')
    return df

@st.cache_data
def load_models():
    # Load pre-trained models (you need to save these from your notebook)
    try:
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('train_data.pkl', 'rb') as f:
            train = pickle.load(f)
        return tfidf, train
    except:
        st.warning("Models not found. Using live computation.")
        return None, None

# Load data
df = load_data()
tfidf, train_df = load_models()

st.title("🎓 Online Course Recommendation System")
st.markdown("---")

# Sidebar for user input
st.sidebar.header("🔍 Search Preferences")
user_id = st.sidebar.number_input("Enter User ID (1-49999)", min_value=1, max_value=49999, value=15796)
n_recommendations = st.sidebar.slider("Number of Recommendations", 5, 15, 5)
difficulty = st.sidebar.multiselect("Difficulty Level", 
                                   ['Beginner', 'Intermediate', 'Advanced'], 
                                   default=['Beginner', 'Intermediate'])
max_price = st.sidebar.slider("Max Price ($)", 0, 500, 300)
min_rating = st.sidebar.slider("Min Rating", 1.0, 5.0, 3.5)

# Recommendation functions (from your notebook)
global_mean = train_df['rating'].mean() if train_df is not None else df['rating'].mean()
user_bias = train_df.groupby('userid')['rating'].mean().subtract(global_mean) if train_df is not None else pd.Series()
item_bias = train_df.groupby('courseid')['rating'].mean().subtract(global_mean) if train_df is not None else pd.Series()

def content_predict(row):
    """Content-based prediction using TF-IDF similarity"""
    if tfidf is None:
        return df['rating'].mean()
    
    course_features = tfidf.transform([f"{row['instructor']} {row['coursename']}"])
    course_similarities = cosine_similarity(course_features, tfidf).flatten()
    similar_ratings = train_df['rating'] * course_similarities[:len(train_df)]
    return similar_ratings.mean() if len(similar_ratings[similar_ratings > 0]) > 0 else global_mean

def cf_predict(row):
    """Collaborative filtering prediction"""
    bu = user_bias.get(row['userid'], 0)
    bi = item_bias.get(row['courseid'], 0)
    return global_mean + bu + bi

def hybrid_predict(row):
    """Hybrid recommendation score"""
    return 0.5 * cf_predict(row) + 0.5 * content_predict(row)

def get_top_n_recommendations(userid, n=5):
    """Get top N recommendations for user"""
    seen = set(df[df['userid'] == userid]['courseid'].unique())
    candidates = df[~df['courseid'].isin(seen)]['courseid'].unique()
    
    scores = []
    for course_id in candidates[:100]:  # Limit for performance
        row = pd.Series({'userid': userid, 'courseid': course_id})
        score = hybrid_predict(row)
        scores.append((course_id, score))
    
    # Filter by user preferences
    top_courses = sorted(scores, key=lambda x: x[1], reverse=True)[:n*3]
    
    recommendations = []
    for course_id, score in top_courses:
        course_info = df[df['courseid'] == course_id].iloc[0]
        if (course_info['difficultylevel'] in difficulty or 
            not difficulty) and course_info['courseprice'] <= max_price and course_info['rating'] >= min_rating:
            recommendations.append({
                'courseid': course_id,
                'coursename': course_info['coursename'],
                'instructor': course_info['instructor'],
                'rating': course_info['rating'],
                'price': course_info['courseprice'],
                'duration': course_info['coursedurationhours'],
                'enrollments': course_info['enrollmentnumbers'],
                'score': score
            })
            if len(recommendations) >= n:
                break
    
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)

# Main recommendation section
col1, col2 = st.columns([1, 3])

with col1:
    st.metric("Total Courses", len(df['courseid'].unique()))
    st.metric("Total Users", len(df['userid'].unique()))
    st.metric("Avg Rating", f"{df['rating'].mean():.2f}")

with col2:
    st.subheader(f"Recommendations for User #{user_id}")
    
    recommendations = get_top_n_recommendations(user_id, n_recommendations)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"**{i}. {rec['coursename']}** ⭐{rec['rating']:.1f}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price", f"${rec['price']:.2f}")
                    st.metric("Duration", f"{rec['duration']:.1f}h")
                with col2:
                    st.metric("Enrollments", f"{rec['enrollments']:,.0f}")
                    st.write(f"**Instructor:** {rec['coursename']}")
                with col3:
                    st.metric("Our Score", f"{rec['score']:.3f}")
    else:
        st.warning("No recommendations match your criteria. Try adjusting filters!")

# Dataset overview
if st.checkbox("📊 Dataset Overview"):
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Avg Course Price", f"${df['courseprice'].mean():.2f}")
        st.metric("Courses with Certification", f"{(df['certificationoffered'] == 'Yes').sum()}")
    
    with col2:
        diff_dist = df['difficultylevel'].value_counts()
        st.write("**Difficulty Distribution**")
        st.bar_chart(diff_dist)

# Top courses
st.subheader("🏆 Top Rated Courses")
top_courses = df.nlargest(10, 'rating')[['coursename', 'instructor', 'rating', 'enrollmentnumbers']]
st.dataframe(top_courses.style.format({
    'rating': '{:.1f}',
    'enrollmentnumbers': '{:,}'
}), use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using your EDA & Hybrid Recommendation Model [file:1]")
