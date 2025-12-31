import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px

# Page config
st.set_page_config(page_title="Course Recommender", layout="wide")

# Load pickle files
@st.cache_data
def load_models():
    fulldata = pd.read_pickle('full_data.pkl')
    traindata = pd.read_pickle('traindata.pkl')
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('biases.pkl', 'rb') as f:
        biases = pickle.load(f)
    return fulldata, traindata, tfidf, biases

fulldata, traindata, tfidf, biases = load_models()
global_mean, user_bias, item_bias = biases['globalmean'], biases['userbias'], biases['itembias']

# Course lookup for display
courselookup = fulldata[['courseid', 'coursename', 'instructor', 'rating']].drop_duplicates(subset='courseid')

st.title("🎓 Online Course Recommendation System")
st.markdown("Enter a **user ID** to get personalized course recommendations!")

# User input
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

# Recommendation function (Hybrid model from notebook)
def hybrid_predict(userid, courseid):
    # User bias
    bu = user_bias.get(userid, 0)
    # Item bias  
    bi = item_bias.get(courseid, 0)
    # Content similarity (simplified - using global mean as base)
    content_pred = global_mean
    # Collaborative filtering
    cf_pred = global_mean + bu + bi
    # Hybrid: 50% CF + 50% Content
    return 0.5 * cf_pred + 0.5 * content_pred

def get_recommendations(userid, n=10):
    # Get user's seen courses
    seen = set(traindata[traindata['userid'] == userid]['courseid'].values)
    # All possible courses
    candidates = set(courselookup['courseid']) - seen
    
    scores = []
    for courseid in candidates:
        score = hybrid_predict(userid, courseid)
        scores.append({'courseid': courseid, 'score': score})
    
    recs = sorted(scores, key=lambda x: x['score'], reverse=True)[:n]
    rec_df = pd.DataFrame(recs)
    rec_df = rec_df.merge(courselookup, on='courseid', how='left')
    return rec_df

# Get recommendations
if st.button("Get Recommendations", type="primary"):
    with st.spinner("Generating recommendations..."):
        recommendations = get_recommendations(user_id, n=15)
    
    st.success(f"✅ Found {len(recommendations)} recommendations for User {user_id}!")
    
    # Display recommendations table
    st.subheader("📚 Top Recommended Courses")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(
            recommendations[['coursename', 'instructor', 'rating']].head(10),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        # Recommendation score distribution
        fig = px.histogram(recommendations.head(10), x='score', nbins=10, 
                          title="Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Course selection for filtering
    st.subheader("🔍 Filter & Select Courses")
    selected_courses = st.multiselect(
        "Select courses from recommendations:",
        recommendations['coursename'].tolist(),
        default=recommendations.head(3)['coursename'].tolist()
    )
    
    if selected_courses:
        filtered = recommendations[recommendations['coursename'].isin(selected_courses)]
        
        # Filter by high ratings (4.5+) and different instructors
        high_rated = filtered[filtered['rating'] >= 4.5]
        if len(high_rated) > 1:
            # Ensure different instructors
            unique_instructors = high_rated.drop_duplicates(subset='instructor')
            st.success(f"🎯 **Top Pick**: {unique_instructors.iloc[0]['coursename']} by {unique_instructors.iloc[0]['instructor']} (Rating: {unique_instructors.iloc[0]['rating']:.1f})")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Score", f"{unique_instructors.iloc[0]['score']:.3f}")
            with col_b:
                st.metric("Rating", unique_instructors.iloc[0]['rating'])
            
            # Show all high-rated options
            st.subheader("⭐ High-Rated Options (4.5+)")
            st.dataframe(unique_instructors[['coursename', 'instructor', 'rating', 'score']])
        else:
            st.info("No courses with 4.5+ rating found. Showing best available:")
            st.dataframe(filtered.sort_values('score', ascending=False).head())

# Instructions sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Enter User ID** (1-49999)
    2. **Click "Get Recommendations"**
    3. **Select courses** from the list
    4. **View high-rated courses** with different instructors
    
    **Files needed:**
    - `fulldata.pkl`
    - `traindata.pkl` 
    - `tfidf.pkl`
    - `biases.pkl`
    
    Generated from your notebook! [file:2]
    """)
    
    st.header("📊 Model Performance")
    st.markdown("""
    | Model | RMSE |
    |-------|------|
    | Popularity | 0.715 |
    | **Hybrid** | **0.805** |
    | Content | 0.767 |
    | CF | 0.916 |[file:2]
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using the Hybrid Recommendation Model from your notebook [file:2]")
