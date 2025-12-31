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

def get_recommendations(user_id, n=20, max_price=500, min_rating=3.0):
    """Get EXACTLY user's unseen courses ranked by hybrid score"""
    # Get courses user 15796 has NOT taken
    user_courses = set(df[df[user_col] == user_id][course_col].unique())
    candidates = df[~df[course_col].isin(user_courses)].copy()
    
    # Filter by price and rating
    candidates = candidates[candidates['rating'] >= min_rating]
    candidates = candidates[candidates[price_col] <= max_price]
    
    if candidates.empty:
        return pd.DataFrame()
    
    # Calculate hybrid scores for ALL candidates
    candidates['score'] = candidates[course_col].apply(lambda x: hybrid_score(user_id, x))
    
    # Sort by score (highest first) and take top N
    recommendations = candidates.nlargest(n, 'score')
    
    # Reorder columns for clean display
    display_cols = [course_col, 'score', name_col, instructor_col, 'rating', price_col]
    return recommendations[display_cols]

# === UI ===
st.title("🎓 Online Course Recommendation System")
st.info(f"🔍 Using columns: User={user_col}, Course={course_col}, Name={name_col}")

st.sidebar.header("🔍 User Controls")
user_id = st.sidebar.number_input("👤 Enter User ID", 1, 49999, 15796, help="User 15796 gets personalized recommendations")
n_recs = st.sidebar.slider("📊 Show Top Recommendations", 0, 20, 10, help="0-20 courses for this user")
max_price = st.sidebar.slider("💰 Max Price ($)", 0, 500, 400)
min_rating = st.sidebar.slider("⭐ Min Rating", 1.0, 5.0, 3.5)

# Stats
col1, col2 = st.columns(2)
col1.metric("📚 Total Courses", len(df[course_col].unique()))
col1.metric("👥 Total Users", len(df[user_col].unique()))
col2.metric("⭐ Avg Rating", f"{df['rating'].mean():.2f}")
col2.metric("💰 Avg Price", f"${df[price_col].mean():.0f}")

st.markdown("---")

# Recommendations Table + Expandable Cards
st.subheader(f"🔥 Top Recommendations for User #{user_id}")
recommendations = get_recommendations(user_id, n_recs, max_price, min_rating)

if not recommendations.empty:
    # Show table first (like your output)
    st.dataframe(
        recommendations.style.format({
            'score': '{:.3f}',
            'rating': '{:.1f}',
            price_col: '${:.0f}'
        }).background_gradient(subset=['score'], cmap='viridis'),
        use_container_width=True,
        hide_index=False
    )
    
    st.markdown("---")
    
    # Expandable detailed cards
    for i, (_, rec) in enumerate(recommendations.iterrows(), 0):  # Starts from 0
        if i >= n_recs:
            break
            
        with st.expander(f"**{i}**. {rec[name_col]} ⭐{rec['rating']:.1f} | 💰${rec[price_col]:.0f} | 🎯Score: {rec['score']:.3f}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**👨‍🏫 Instructor:** {rec[instructor_col]}")
                st.markdown(f"**📦 Course ID:** {rec[course_col]}")
                if 'coursedurationhours' in df.columns:
                    st.markdown(f"**⏱️ Duration:** {df[df[course_col]==rec[course_col]]['coursedurationhours'].iloc[0]:.1f}h")
                if 'enrollmentnumbers' in df.columns:
                    st.markdown(f"**👥 Enrollments:** {df[df[course_col]==rec[course_col]]['enrollmentnumbers'].iloc[0]:,.0f}")
            with col2:
                st.progress(rec['rating'] / 5.0)
                st.metric("Hybrid Score", f"{rec['score']:.3f}")
else:
    st.warning(f"😅 No recommendations found for User #{user_id}. Try adjusting filters!")

# User's taken courses
st.markdown("---")
with st.expander(f"📜 Courses already taken by User #{user_id}"):
    user_taken = df[df[user_col] == user_id][[course_col, name_col, instructor_col, 'rating']].head(10)
    st.dataframe(user_taken, use_container_width=True)

st.markdown("---")
st.success(f"✅ Personalized recommendations for User #{user_id} - Hybrid Model (60% CF + 40% Content)")

