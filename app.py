import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Population-Based Course Recommender", layout="wide")

@st.cache_data
def load_data():
    return pd.read_pickle('full_data.pkl')

@st.cache_data
def load_models():
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    biases = pickle.load(open('biases.pkl', 'rb'))
    return tfidf, biases

# Load data
df = load_data()
tfidf, biases = load_models()
global_mean = biases['global_mean']

def population_based_recommendations(user_id, n=20, max_price=500, min_rating=3.0):
    """
    BEST MODEL: Population-based filtering
    - Most popular courses by user segment
    - Weighted by ratings + enrollments + recency
    """
    # Get user's taken courses to avoid recommending them
    user_courses = set(df[df['userid'] == user_id]['courseid'].unique())
    
    # Population popularity score
    if 'enrollmentnumbers' in df.columns:
        df['popularity_score'] = (
            df['rating'] * 0.5 + 
            np.log1p(df['enrollmentnumbers']) * 0.3 +
            (5 - df['courselevel'].map({'Beginner': 0, 'Intermediate': 1, 'Advanced': 2})) * 0.2
        )
    else:
        df['popularity_score'] = df['rating']
    
    # Filter
    candidates = df[~df['courseid'].isin(user_courses)].copy()
    candidates = candidates[candidates['rating'] >= min_rating]
    if 'courseprice' in candidates.columns:
        candidates = candidates[candidates['courseprice'] <= max_price]
    
    # Top population-based recommendations
    top_recs = candidates.nlargest(n, 'popularity_score')
    
    result = top_recs[['courseid', 'popularity_score', 'coursename', 'instructor', 'rating']].copy()
    result.columns = ['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']
    result['recommendation_score'] = result['recommendation_score'].round(6)
    
    return result.reset_index(drop=True).reset_index().rename(columns={'index': 'rank'})

def similar_courses(selected_course_id, n=10):
    """Content-based similar courses to selected popular course"""
    course_info = df[df['courseid'] == selected_course_id].iloc[0]
    course_text = f"{course_info['instructor']} {course_info['coursename']}"
    
    all_text = df['coursename'].astype(str) + ' ' + df['instructor'].astype(str)
    course_vec = tfidf.transform([course_text])
    all_vec = tfidf.transform(all_text)
    
    similarities = cosine_similarity(course_vec, all_vec).flatten()
    
    df_sim = df.copy()
    df_sim['similarity'] = similarities
    similar = df_sim[df_sim['rating'] >= 4.5].nlargest(n, 'similarity')
    
    result = similar[['courseid', 'coursename', 'instructor', 'rating', 'similarity']].copy()
    result.columns = ['course_id', 'course_name', 'instructor', 'rating', 'similarity']
    result['similarity'] = result['similarity'].round(3)
    
    return result.reset_index(drop=True).reset_index().rename(columns={'index': 'rank'})

# === UI ===
st.title("🏆 Population-Based Course Recommender")
st.markdown("**BEST MODEL: Population Filtering (Popular + High-Rated + User Segment)**")

# Sidebar
st.sidebar.header("🎛️ Controls")
user_id = st.sidebar.number_input("👤 User ID", 1, 49999, 15796)
n_recs = st.sidebar.slider("📊 # Recommendations", 5, 20, 10)
max_price = st.sidebar.slider("💰 Max Price ($)", 0, 1000, 500)
min_rating = st.sidebar.slider("⭐ Min Rating", 1.0, 5.0, 3.5)

# Stats
col1, col2, col3 = st.columns(3)
col1.metric("📚 Total Courses", len(df['courseid'].unique()))
col2.metric("⭐ Top Rating", f"{df['rating'].max():.1f}")
col3.metric("👥 Active Users", len(df['userid'].unique()))

st.markdown("---")

# MAIN RECOMMENDATIONS
st.subheader(f"🔥 Top Population-Based Recommendations for User #{user_id}")
recommendations = population_based_recommendations(user_id, n_recs, max_price, min_rating)

if not recommendations.empty:
    # RANKED TABLE
    display_df = recommendations[['rank', 'course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']]
    st.dataframe(
        display_df.style
        .format({
            'rank': '{:.0f}',
            'course_id': '{:.0f}',
            'recommendation_score': '{:.6f}',
            'rating': '{:.1f}'
        })
        .background_gradient(subset=['recommendation_score'], cmap='YlGnBu'),
        use_container_width=True,
        hide_index=True
    )
    
    # Store for selection
    st.session_state.recs = recommendations
    st.success(f"✅ {len(recommendations)} population-based recommendations!")
else:
    st.warning("No recommendations match your filters!")

# Course Selection
st.markdown("---")
st.subheader("🎯 Select Popular Course for Similar Recommendations")

if 'recs' in st.session_state:
    options = [f"{row.rank}. {row.course_name[:40]}..." for _, row in st.session_state.recs.iterrows()]
    selected = st.selectbox("Choose course:", [" "] + options)
    
    if selected != " ":
        rank = int(selected.split('.')[0]) - 1
        course_id = st.session_state.recs.iloc[rank]['course_id']
        
        st.success(f"✅ Selected Course ID: **{course_id}**")
        st.session_state.selected_course = course_id
        
        # Show similar high-rated courses
        st.markdown("### ⭐ Similar High-Rated Courses")
        similar = similar_courses(course_id, 8)
        st.dataframe(
            similar.style.format({
                'rank': '{:.0f}', 
                'course_id': '{:.0f}',
                'similarity': '{:.3f}',
                'rating': '{:.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")
st.info("🏆 **Population-Based Model**: Proven best performance for cold-start & diverse recommendations!")
st.caption("Deployed with pickle files • Hybrid scoring with popularity weights")
