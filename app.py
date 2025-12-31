import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Online Course Recommendation System")
st.markdown("Discover the best courses tailored just for you based on our Hybrid Recommendation Engine.")

# --- LOAD DATA AND MODELS ---
@st.cache_resource
def load_models():
    try:
        # Load the original full dataset for metadata
        full_data = pd.read_pickle('full_data.pkl')
        
        # Load the standardized training data
        train_data = pd.read_pickle('train_data.pkl')
        
        # Load the biases (Global Mean, User Bias, Item Bias)
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
            
        # Load TF-IDF (optional, but loaded as it was saved)
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
            
        return full_data, train_data, biases, tfidf
    except FileNotFoundError as e:
        st.error(f"Missing model files: {e.filename}. Please run the training notebook first.")
        return None, None, None, None

df, train, biases, tfidf = load_models()

if df is not None:
    # Extract model parameters
    global_mean = biases['global_mean']
    user_bias = biases['user_bias']
    item_bias = biases['item_bias']

    # --- RECOMMENDATION LOGIC ---
    def cf_predict(user_id, course_id):
        """Collaborative Filtering Prediction: Mean + User Bias + Item Bias"""
        bu = user_bias.get(user_id, 0)
        bi = item_bias.get(course_id, 0)
        return global_mean + bu + bi

    def content_predict(course_id):
        """Content-Based Prediction: Average rating for the course in training data"""
        ratings = train[train['courseid'] == course_id]['rating']
        return ratings.mean() if not ratings.empty else global_mean

    def hybrid_predict(user_id, course_id):
        """Hybrid Prediction: 50% CF + 50% Content"""
        return 0.5 * cf_predict(user_id, course_id) + 0.5 * content_predict(course_id)

    def get_top_n(user_id, n=5):
        """Finds top N courses the user hasn't seen yet"""
        # Courses user has already interacted with
        seen_courses = train[train['userid'] == user_id]['courseid'].unique()
        
        # All available courses in the system
        all_courses = train['courseid'].unique()
        
        # Candidates are courses not yet seen by the user
        candidates = [c for c in all_courses if c not in seen_courses]
        
        if not candidates:
            return []
            
        scores = []
        for cid in candidates:
            score = hybrid_predict(user_id, cid)
            scores.append((cid, score))
            
        # Sort by prediction score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    # --- SIDEBAR ---
    st.sidebar.header("User Selection")
    user_list = sorted(train['userid'].unique())
    selected_user = st.sidebar.selectbox("Select a User ID to get recommendations:", user_list)
    num_recommendations = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

    # --- MAIN UI ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Recommendations for User: {selected_user}")
        if st.button("Generate Recommendations"):
            with st.spinner('Calculating scores...'):
                recommendations = get_top_n(selected_user, num_recommendations)
            
            if recommendations:
                # Prepare a lookup table for display
                # We use the original dataframe 'df' to get course titles and instructors
                # Note: 'df' column names are from original Excel, 'train' names are standardized
                course_lookup = df[['course_id', 'course_name', 'instructor', 'rating', 'difficulty_level', 'course_price']].drop_duplicates('course_id')
                
                for rank, (cid, score) in enumerate(recommendations, 1):
                    details = course_lookup[course_lookup['course_id'] == cid].iloc[0]
                    
                    with st.expander(f"#{rank}: {details['course_name']}", expanded=True):
                        c1, c2, c3 = st.columns([2, 1, 1])
                        with c1:
                            st.write(f"**Instructor:** {details['instructor']}")
                            st.write(f"**Level:** {details['difficulty_level']}")
                        with c2:
                            st.write(f"**Price:** ${details['course_price']}")
                            st.write(f"**Avg Rating:** {details['rating']} ⭐")
                        with c3:
                            st.metric("Match Score", f"{score:.2f}")
            else:
                st.info("This user has already seen all available courses.")

    with col2:
        st.subheader("User Activity")
        user_history = train[train['userid'] == selected_user]
        st.write(f"User has completed **{len(user_history)}** courses.")
        
        # Merge with df to show names of courses already taken
        history_names = user_history.merge(
            df[['course_id', 'course_name']].drop_duplicates(), 
            left_on='courseid', right_on='course_id'
        )
        st.dataframe(history_names[['course_name', 'rating']], use_container_width=True, hide_index=True)

    # --- SEARCH FEATURE ---
    st.divider()
    st.subheader("🔍 Explore All Courses")
    search_query = st.text_input("Search courses by name or instructor:")
    if search_query:
        search_results = df[
            df['course_name'].str.contains(search_query, case=False, na=False) | 
            df['instructor'].str.contains(search_query, case=False, na=False)
        ].drop_duplicates('course_id')
        st.dataframe(search_results[['course_name', 'instructor', 'rating', 'difficulty_level', 'course_price']], hide_index=True)

else:
    st.warning("Please ensure the following files are in the directory: `full_data.pkl`, `train_data.pkl`, `biases.pkl`, `tfidf.pkl`.")
