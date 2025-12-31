import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="Course Recommender", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .stTable { font-size: 14px; }
    .main-header { color: #1f4e79; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING WITH ROBUSTNESS ---
@st.cache_resource
def load_assets():
    # List of required files
    files = ['full_data.pkl', 'train_data.pkl', 'biases.pkl', 'tfidf.pkl']
    
    # Check if files are missing or empty (0 bytes)
    for f in files:
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            st.error(f"File **{f}** is missing or corrupted. Please run the model training notebook again.")
            return None

    try:
        full_df = pd.read_pickle('full_data.pkl')
        train_df = pd.read_pickle('train_data.pkl')
        
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
            
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
            
        # Metadata lookup table
        lookup = full_df[['course_id', 'course_name', 'instructor', 'rating']].drop_duplicates()
        
        # Prepare TF-IDF matrix for "Further Recommendations"
        full_df['features'] = full_df['course_name'].astype(str) + ' ' + full_df['instructor'].astype(str)
        tfidf_matrix = tfidf.transform(full_df['features'])
        
        return full_df, train_df, biases, tfidf_matrix, lookup
    except Exception as e:
        st.error(f"Error loading system assets: {e}")
        return None

# Initialize assets
assets = load_assets()

if assets:
    df, train, biases, tfidf_matrix, lookup = assets
    
    st.title("🎓 Online Course Recommendation System")
    
    # User Input
    user_id_input = st.number_input("Enter User ID to get Recommendations:", value=15796, step=1)

    if st.button("Generate My Recommendations"):
        # 1. Recommendation Logic (Hybrid: 50% CF + 50% Content)
        g_mean = biases.get('global_mean', 0)
        u_bias = biases.get('user_bias', {})
        i_bias = biases.get('item_bias', {})
        
        seen = train[train['userid'] == user_id_input]['courseid'].unique()
        all_c = train['courseid'].unique()
        candidates = [c for c in all_c if c not in seen]
        
        scores = []
        for cid in candidates:
            # Collaborative Filtering Part
            cf = g_mean + u_bias.get(user_id_input, 0) + i_bias.get(cid, 0)
            # Content Part (Mean rating)
            m_rating = train[train['courseid'] == cid]['rating'].mean()
            # Hybrid
            final_score = 0.5 * cf + 0.5 * m_rating
            scores.append((cid, final_score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        top_n = pd.DataFrame(scores[:5], columns=['course_id', 'recommendation_score'])
        
        # Merge with metadata for the final table
        final_table = top_n.merge(lookup, on='course_id', how='left')
        
        # Formatting: 6 decimal places for the score
        final_table['recommendation_score'] = final_table['recommendation_score'].apply(lambda x: f"{x:.6f}")
        
        st.subheader(f"Top Recommendations for User {user_id_input}")
        
        # Exact column order requested
        display_cols = ['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']
        st.table(final_table[display_cols].reset_index(drop=True))
        
        # Store in session state for the "Further Selection" step
        st.session_state['user_recs'] = final_table

    # --- FURTHER SELECTION FEATURE ---
    if 'user_recs' in st.session_state:
        st.divider()
        st.subheader("🎯 Refine Your Selection")
        st.info("Because 'Mobile App Development with Swift' has multiple instructors, please select the specific version you are interested in:")
        
        # Create unique labels (Name + Instructor) to distinguish duplicates
        recs = st.session_state['user_recs']
        recs['display_label'] = recs['course_name'] + " (" + recs['instructor'] + ")"
        
        selected_label = st.selectbox(
            "Pick a course to see similar high-rated alternatives:",
            ["Select a course..."] + recs['display_label'].tolist()
        )
        
        if selected_label != "Select a course...":
            # Extract course_id from the selected row in session state
            selected_row = recs[recs['display_label'] == selected_label].iloc[0]
            selected_cid = selected_row['course_id']
            
            # Find the corresponding index in the original dataframe
            idx = df[df['course_id'] == selected_cid].index[0]
            
            # Compute similarity
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            temp_df = df.copy()
            temp_df['similarity'] = sim_scores
            
            # Filter: Exclude current course, High similarity, High rating (>= 4.0)
            discovery = (
                temp_df[temp_df['course_id'] != selected_cid]
                .query("rating >= 4.0")
                .sort_values(by=['similarity', 'rating'], ascending=False)
                .drop_duplicates('course_name')
                .head(5)
            )
            
            st.write(f"**Highly-rated courses similar to '{selected_label}':**")
            st.table(discovery[['course_name', 'instructor', 'rating', 'difficulty_level', 'course_price']])

else:
    st.error("System files not found. Please ensure `full_data.pkl`, `train_data.pkl`, `biases.pkl`, and `tfidf.pkl` are in your app directory.")
