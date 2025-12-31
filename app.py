import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="Course Advisor", layout="wide")

# --- LOAD DATA ---
@st.cache_resource
def load_all_assets():
    try:
        # Load deduplicated metadata and training files
        full_df = pd.read_pickle('full_data.pkl')
        train_df = pd.read_pickle('train_data.pkl')
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
            
        # Prepare similarity matrix for the "further selection" feature
        tfidf_matrix = tfidf.transform(full_df['content'])
        
        return full_df, train_df, biases, tfidf_matrix
    except Exception as e:
        st.error(f"Error loading files: {e}. Please ensure the files exist.")
        return None

assets = load_all_assets()

if assets:
    full_df, train_df, biases, tfidf_matrix = assets
    
    st.title("🎓 Online Course Recommendation System")
    
    # User ID Input
    user_id = st.number_input("Enter User ID:", value=15796, step=1)
    
    if st.button("Generate Recommendations"):
        # 1. Prediction Logic (Hybrid 50/50)
        g_mean = biases['global_mean']
        u_b = biases['user_bias']
        i_b = biases['item_bias']
        
        # Get courses the user hasn't taken
        seen = train_df[train_df['userid'] == user_id]['courseid'].unique()
        all_courses = train_df['courseid'].unique()
        candidates = [c for c in all_courses if c not in seen]
        
        scores = []
        for cid in candidates:
            # Collaborative Filtering Part
            cf = g_mean + u_b.get(user_id, 0) + i_b.get(cid, 0)
            # Item Mean Part
            item_mean = train_df[train_df['courseid'] == cid]['rating'].mean()
            # Hybrid Score
            score = 0.5 * cf + 0.5 * item_mean
            scores.append((cid, score))
            
        # Sort and take top 5
        scores.sort(key=lambda x: x[1], reverse=True)
        top_5 = pd.DataFrame(scores[:5], columns=['course_id', 'recommendation_score'])
        
        # 2. Format the output table
        # Merge with metadata (full_df is already deduplicated)
        final_table = top_5.merge(full_df, left_on='course_id', right_on='course_id', how='left')
        
        # Exact Column Order and Formatting
        final_table['recommendation_score'] = final_table['recommendation_score'].apply(lambda x: f"{x:.6f}")
        
        st.subheader(f"Top Recommendations for User {user_id}")
        # Selecting specific columns for display
        display_df = final_table[['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']]
        st.table(display_df)
        
        # Save to session state for the selection feature
        st.session_state['recommendations'] = final_table

    # --- FURTHER SELECTION FEATURE ---
    if 'recommendations' in st.session_state:
        st.divider()
        st.subheader("🎯 Refine Your Choice")
        st.write("Select a recommended course to see similar high-rated options:")
        
        options = st.session_state['recommendations']['course_name'].tolist()
        selected_course = st.selectbox("Pick a course:", ["Select a course..."] + options)
        
        if selected_course != "Select a course...":
            # Find the ID and index of the selected course
            sel_row = st.session_state['recommendations'][st.session_state['recommendations']['course_name'] == selected_course].iloc[0]
            sel_id = sel_row['course_id']
            
            # Find index in full_df
            idx = full_df[full_df['course_id'] == sel_id].index[0]
            
            # Compute similarity
            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            # Create a temp DF for similarity results
            sim_df = full_df.copy()
            sim_df['sim_score'] = cosine_sim
            
            # Filter for High Ratings (>= 4.0) and sort
            discovery = sim_df[sim_df['course_id'] != sel_id]
            discovery = discovery[discovery['rating'] >= 4.0]
            discovery = discovery.sort_values(by=['sim_score', 'rating'], ascending=False).head(5)
            
            st.write(f"**Based on '{selected_course}', you might also like these top-rated courses:**")
            st.table(discovery[['course_name', 'instructor', 'rating', 'course_price']])
