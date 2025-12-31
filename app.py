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
    user_courses = set(df[df[user_col] == user_id][course_col].unique())
    candidates = df[~df[course_col].isin(user_courses)].copy()
    
    candidates = candidates[candidates['rating'] >= min_rating]
    candidates = candidates[candidates[price_col] <= max_price]
    
    candidates['recommendation_score'] = candidates[course_col].apply(lambda x: hybrid_score(user_id, x))
    top_recs = candidates.nlargest(n, 'recommendation_score')
    
    result = top_recs[[course_col, 'recommendation_score', name_col, instructor_col, 'rating', price_col]].copy()
    result.columns = ['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating', 'price']
    result.reset_index(drop=True, inplace=True)
    result.index += 1  # Start from 1
    return result

def get_high_rating_courses(selected_course_id, n=10):
    selected_info = df[df[course_col] == selected_course_id].iloc[0]
    selected_text = f"{selected_info[instructor_col]} {selected_info[name_col]}"
    
    all_text = df[name_col].astype(str) + ' ' + df[instructor_col].astype(str)
    selected_features = tfidf.transform([selected_text])
    all_features = tfidf.transform(all_text)
    similarities = cosine_similarity(selected_features, all_features).flatten()
    
    df_sim = df.copy()
    df_sim['similarity'] = similarities
    high_rating = df_sim[df_sim['rating'] >= 4.5].nlargest(n, 'similarity')
    
    result = high_rating[[course_col, name_col, instructor_col, 'rating', 'similarity']].copy()
    result.columns = ['course_id', 'course_name', 'instructor', 'rating', 'similarity']
    result['similarity'] = result['similarity'].round(3)
    result.reset_index(drop=True, inplace=True)
    result.index += 1
    return result

# === MAIN UI ===
st.title("🎓 Online Course Recommendation System")

# Sidebar
st.sidebar.header("🔍 Controls")
user_id = st.sidebar.number_input("👤 Enter User ID", 1, 49999, 15796, key="main_user_id")
max_price = st.sidebar.slider("💰 Max Price ($)", 0, 500, 400)
min_rating = st.sidebar.slider("⭐ Min Rating", 1.0, 5.0, 3.0)

# RECOMMENDATIONS SECTION - ALWAYS SHOWS COURSES
st.subheader(f"🔥 Top Recommendations for User #{user_id}")

# Get and display recommendations IMMEDIATELY
recommendations = get_recommendations(user_id, 20, max_price, min_rating)

if not recommendations.empty:
    # Header
    st.markdown("**# | course_id | score | course_name | instructor | rating | price**")
    
    # Show ALL recommendations (1-20)
    for idx, row in recommendations.iterrows():
        course_name_short = (row['course_name'][:28] + "...") if len(row['course_name']) > 28 else row['course_name']
        instructor_short = (row['instructor'][:12] + "...") if len(row['instructor']) > 12 else row['instructor']
        
        st.markdown(f"""
        **{idx}** | **{int(row['course_id'])}** | **{row['recommendation_score']:.3f}** | 
        **{course_name_short}** | **{instructor_short}** | **{row['rating']:.1f}** | **${int(row['price'])}**
        """)
    
    # Store for selection
    st.session_state.recommendations = recommendations
    
    st.success(f"✅ Showing top {len(recommendations)} recommendations for User #{user_id}")
else:
    st.warning("No recommendations found! Try adjusting filters.")

st.markdown("---")

# Course Selection
st.subheader("🎯 Select a Course from Recommendations")
if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
    course_options = []
    for idx, row in st.session_state.recommendations.head(10).iterrows():  # Top 10 only
        course_options.append(f"{int(idx)}. {row['course_name']} (ID: {int(row['course_id'])})")
    
    selected_course = st.selectbox("Choose course:", [""] + course_options, key="course_select")
    
    if selected_course and selected_course != "":
        selected_idx = int(selected_course.split('.')[0])
        selected_course_id = st.session_state.recommendations.iloc[selected_idx-1]['course_id']
        st.session_state.selected_course_id = selected_course_id
        
        st.success(f"✅ Selected Course ID: **{int(selected_course_id)}**")
else:
    st.info("👆 Generate recommendations above first!")

# High Rating Courses
if 'selected_course_id' in st.session_state:
    st.subheader("⭐ High Rating Courses (Similar to Selected)")
    high_rating_courses = get_high_rating_courses(st.session_state.selected_course_id, 10)
    
    if not high_rating_courses.empty:
        st.markdown("**# | course_id | course_name | instructor | rating | similarity**")
        
        for idx, row in high_rating_courses.iterrows():
            course_name_short = (row['course_name'][:28] + "...") if len(row['course_name']) > 28 else row['course_name']
            instructor_short = (row['instructor'][:12] + "...") if len(row['instructor']) > 12 else row['instructor']
            
            st.markdown(f"""
            **{idx}** | **{int(row['course_id'])}** | **{course_name_short}** | 
            **{instructor_short}** | **{row['rating']:.1f}** | **{row['similarity']}**
            """)
    else:
        st.warning("No high rating similar courses found!")

st.markdown("---")
st.caption("✅ Recommendations ALWAYS show for User ID input!")
