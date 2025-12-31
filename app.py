import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Page config
st.set_page_config(page_title="Course Recommender", layout="wide")

# Smart pickle file loader (handles both naming conventions)
@st.cache_data
def load_models():
    # Try both fulldata.pkl and full_data.pkl
    fulldata_file = 'fulldata.pkl' if os.path.exists('fulldata.pkl') else 'full_data.pkl'
    
    try:
        fulldata = pd.read_pickle(fulldata_file)
        traindata = pd.read_pickle('traindata.pkl')
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
        
        st.sidebar.success(f"Loaded: {fulldata_file}, traindata.pkl, tfidf.pkl, biases.pkl")
        return fulldata, traindata, tfidf, biases
        
    except FileNotFoundError as e:
        st.error(f" Missing file: {e}")
        st.error("**Required files:**")
        st.error("```
full_data.pkl (or fulldata.pkl)
traindata.pkl
tfidf.pkl  
biases.pkl
        ```")
        st.stop()

# Load models
fulldata, traindata, tfidf, biases = load_models()
global_mean, user_bias, item_bias = biases['globalmean'], biases['userbias'], biases['itembias']

# Course lookup
courselookup = fulldata[['courseid', 'coursename', 'instructor', 'rating']].drop_duplicates(subset='courseid')

st.title(" Online Course Recommendation System")
st.markdown("Enter a **user ID** to get personalized course recommendations!")

# User input
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

# Hybrid prediction function
def hybrid_predict(userid, courseid):
    bu = user_bias.get(userid, 0)
    bi = item_bias.get(courseid, 0)
    return global_mean + bu + bi

def get_recommendations(userid, n=15):
    seen = set(traindata[traindata['userid'] == userid]['courseid'].values)
    candidates = set(courselookup['courseid']) - seen
    
    scores = []
    for courseid in list(candidates)[:100]:  # Performance limit
        score = hybrid_predict(userid, courseid)
        scores.append({'courseid': courseid, 'score': score})
    
    recs = sorted(scores, key=lambda x: x['score'], reverse=True)[:n]
    rec_df = pd.DataFrame(recs)
    return rec_df.merge(courselookup, on='courseid', how='left')

# Recommendations button
if st.button(" Get Recommendations", type="primary"):
    with st.spinner("Generating personalized recommendations..."):
        recommendations = get_recommendations(user_id)
    
    st.success(f"{len(recommendations)} recommendations for User {user_id}!")
    
    # Top recommendations table
    st.subheader(" Top Recommendations")
    st.dataframe(
        recommendations[['coursename', 'instructor', 'rating', 'score']].head(10),
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("Pred Score", format="%.3f"),
            "rating": st.column_config.NumberColumn("Avg Rating", format="%.1f")
        }
    )
    
    # Course selection & filtering
    st.subheader(" Select Courses for High-Rating Filter")
    selected_courses = st.multiselect(
        "Pick from recommendations:",
        recommendations['coursename'].head(15).tolist(),
        default=recommendations.head(3)['coursename'].tolist()
    )
    
    if selected_courses:
        filtered = recommendations[recommendations['coursename'].isin(selected_courses)]
        
        # Filter: High rating (4.5+) + Different instructors
        high_rated = filtered[filtered['rating'] >= 4.5].drop_duplicates(subset='instructor')
        
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            st.balloons()
            st.success(f" **Top Pick**: *{best['coursename']}* by **{best['instructor']}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(" Predicted Score", f"{best['score']:.3f}")
            col2.metric(" Average Rating", best['rating'])
            col3.metric(" Instructor", best['instructor'])
            
            st.subheader(" All High-Rated Options (4.5+)")
            st.dataframe(high_rated[['coursename', 'instructor', 'rating', 'score']])
        else:
            st.warning(" No 4.5+ rated courses found. Top options:")
            st.dataframe(filtered.sort_values('score', ascending=False).head())

# Sidebar info
with st.sidebar:
    st.header("📋 Quick Guide")
    st.markdown("""
    1. Enter **User ID** (1-49999)
    2. Click **Get Recommendations**
    3. Select courses from list
    4. Get **high-rated (4.5+) courses** with **different instructors**
    """)
    
    st.header(" File Status")
    st.markdown(f"""
    ** Supports both:**
    - `full_data.pkl` ← **Your preference**
    - `fulldata.pkl` 
    - `traindata.pkl`
    - `tfidf.pkl`
    - `biases.pkl`
    """)
    
    st.header(" Setup")
    st.code("""
pip install streamlit pandas numpy scikit-learn
streamlit run app.py
    """, language="bash")

st.markdown("---")
st.markdown("*Powered by your Hybrid Model (CF + Biases) from the notebook [file:2]*")
