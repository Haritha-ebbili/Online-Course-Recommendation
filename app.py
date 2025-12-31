import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Course Recommender", layout="wide")

@st.cache_data
def load_data():
    """Load full dataset"""
    return pd.read_pickle('full_data.pkl')

@st.cache_data  
def load_models():
    """Load all ML models"""
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    train_df = pd.read_pickle('train_data.pkl')
    biases = pickle.load(open('biases.pkl', 'rb'))
    return tfidf, train_df, biases

# Load everything
try:
    df = load_data()
    tfidf, train_df, biases = load_models()
    global_mean = biases['global_mean']
    user_bias = biases['user_bias']
    item_bias = biases['item_bias']
    
    st.session_state.loaded = True
    st.sidebar.success("✅ Models Loaded!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

def content_based_score(course_id):
    """Content-based similarity score"""
    course_info = df[df['courseid'] == course_id].iloc[0]
    course_text = f"{course_info['instructor']} {course_info['coursename']}"
    
    course_vec = tfidf.transform([course_text])
    all_text = df['coursename'].astype(str) + ' ' + df['instructor'].astype(str)
    all_vec = tfidf.transform(all_text)
    
    sim_scores = cosine_similarity(course_vec, all_vec).flatten()
    weights = sim_scores[:len(train_df)]
    ratings = train_df['rating'].values
    
    valid = weights > 0.01
    if valid.sum() > 0:
        return np.average(ratings[valid], weights=weights[valid])
    return global_mean

def collaborative_filtering_score(user_id, course_id):
    """CF score using biases"""
    bu = user_bias.get(user_id, 0)
    bi = item_bias.get(course_id, 0)
    return global_mean + bu + bi

def hybrid_score(user_id, course_id):
    """60% CF + 40% Content"""
    cf_score = collaborative_filtering_score(user_id, course_id)
    content_score = content_based_score(course_id)
    return 0.6 * cf_score + 0.4 * content_score

def get_recommendations(user_id, n=10, max_price=500, min_rating=3.0):
    """Get personalized recommendations"""
    # Courses user hasn't taken
    user_courses = set(df[df['userid'] == user_id]['courseid'].unique())
    candidates = df[~df['courseid'].isin(user_courses)].copy()
    
    # Apply filters
    candidates = candidates[candidates['rating'] >= min_rating]
    if 'courseprice' in candidates.columns:
        candidates = candidates[candidates['courseprice'] <= max_price]
    
    # Calculate hybrid scores
    candidates['rec_score'] = candidates['courseid'].apply(
        lambda x: hybrid_score(user_id, x)
    )
    
    # Top N recommendations
    top_recs = candidates.nlargest(n, 'rec_score')
    result = top_recs[['courseid', 'rec_score', 'coursename', 'instructor', 'rating']].copy()
    result.columns = ['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']
    return result.reset_index(drop=True)

# === DASHBOARD ===
st.title("🎓 Advanced Course Recommendation System")
st.markdown("**Hybrid Model: 60% Collaborative Filtering + 40% Content-Based**")

# Sidebar controls
st.sidebar.header("🔧 Controls")
user_id = st.sidebar.number_input("User ID", 1, 49999, 15796)
n_recs = st.sidebar.slider("Recommendations", 5, 20, 10)
max_price = st.sidebar.slider("Max Price", 0, 1000, 500)
min_rating = st.sidebar.slider("Min Rating", 1.0, 5.0, 3.0)

# Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("📚 Total Courses", len(df['courseid'].unique()))
col2.metric("👥 Total Users", len(df['userid'].unique()))
col3.metric("⭐ Avg Rating", f"{df['rating'].mean():.2f}")
col4.metric("💰 Avg Price", f"${df.get('courseprice', pd.Series([0])).mean():.0f}")

# Recommendations
st.markdown(f"## 🔥 Top Recommendations for User #{user_id}")
recommendations = get_recommendations(user_id, n_recs, max_price, min_rating)

if not recommendations.empty:
    # Beautiful table
    st.dataframe(
        recommendations.style
        .format({'recommendation_score': '{:.6f}', 'rating': '{:.1f}'})
        .background_gradient(subset=['recommendation_score'], cmap='YlOrRd'),
        use_container_width=True,
        hide_index=False
    )
    
    # Store for further use
    st.session_state.recommendations = recommendations
else:
    st.warning("No recommendations found!")

# Success message
st.markdown("---")
st.success(f"✅ {len(recommendations)} personalized recommendations generated!")
st.caption("Deployed on Streamlit Cloud with pickle files")
