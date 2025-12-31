import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="Course Recommender", layout="wide")

st.markdown("""
    <style>
    .stTable { font-size: 14px; }
    .main-title { color: #2E4053; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_data():
    try:
        # Load assets generated from your notebook
        full_df = pd.read_pickle('full_data.pkl')
        train_df = pd.read_pickle('train_data.pkl')
        
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
            
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
            
        # Standardize lookup metadata
        lookup = full_df[['course_id', 'course_name', 'instructor', 'rating']].drop_duplicates('course_id')
        
        # Prepare for Content Similarity
        full_df['content_features'] = full_df['course_name'].astype(str) + ' ' + full_df['instructor'].astype(str)
        tfidf_matrix = tfidf.transform(full_df['content_features'])
        
        return full_df, train_df, biases, tfidf_matrix, lookup
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

df, train, biases, tfidf_matrix, lookup = load_data()

# --- PREDICTION LOGIC ---
def get_hybrid_score(user_id, course_id, global_mean, user_bias, item_bias, train_df):
    # Collaborative Filtering Part (Handling potential missing keys)
    bu = user_bias.get(user_id, 0)
    bi = item_bias.get(course_id, 0)
    cf_pred = global_mean + bu + bi
    
    # Content Part (Average rating of the specific course)
    course_ratings = train_df[train_df['courseid'] == course_id]['rating']
    content_pred = course_ratings.mean() if not course_ratings.empty else global_mean
    
    # Hybrid Score: 50% CF + 50% Content
    return 0.5 * cf_pred + 0.5 * content_pred

def get_recommendations(user_id, n=5):
    # Check if 'biases' is actually a dictionary and contains required keys
    if not isinstance(biases, dict) or 'global_mean' not in biases:
        st.error("Bias dictionary is corrupted. Please re-run the Model Building notebook.")
        return []

    g_mean = biases['global_mean']
    u_bias = biases.get('user_bias', {})
    i_bias = biases.get('item_bias', {})
    
    # Find courses not yet taken by user
    seen = train[train['userid'] == user_id]['courseid'].unique()
    all_courses = train['courseid'].unique()
    candidates = [c for c in all_courses if c not in seen]
    
    scores = []
    for cid in candidates:
        score = get_hybrid_score(user_id, cid, g_mean, u_bias, i_bias, train)
        scores.append((cid, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]

# --- UI INTERFACE ---
st.title("🎓 Online Course Recommendation System")

user_input = st.number_input("Enter User ID:", value=15796, step=1)

if st.button("Generate Recommendations"):
    # Clear previous session state if user ID changes
    if 'last_user' in st.session_state and st.session_state['last_user'] != user_input:
        if 'recs_visible' in st.session_state: del st.session_state['recs_visible']
    
    st.session_state['last_user'] = user_input
    
    recs = get_recommendations(user_input)
    
    if recs:
        st.subheader(f"Top Recommendations for User {user_input}")
        
        # Build the exact table structure requested
        rec_df = pd.DataFrame(recs, columns=['course_id', 'recommendation_score'])
        final_table = rec_df.merge(lookup, on='course_id', how='left')
        
        # Format score to match your 6-decimal requirement
        final_table['recommendation_score'] = final_table['recommendation_score'].astype(float).map('{:.6f}'.format)
        
        # Reorder columns to match your exact output
        display_cols = ['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']
        st.table(final_table[display_cols].reset_index(drop=True))
        
        st.session_state['recs_visible'] = final_table
    else:
        st.warning(f"No new recommendations available for User {user_input}.")

# --- DYNAMIC EXPLORER FEATURE ---
if 'recs_visible' in st.session_state:
    st.markdown("---")
    st.subheader("🔍 Explore Further")
    st.info("Select a course from your recommendations to find similar high-rated alternatives.")
    
    selected_name = st.selectbox(
        "Which course interests you most?",
        ["Select a course..."] + st.session_state['recs_visible']['course_name'].tolist()
    )
    
    if selected_name != "Select a course...":
        # Recommendation based on Content Similarity
        idx = df[df['course_name'] == selected_name].index[0]
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        sim_df = df.copy()
        sim_df['similarity'] = sim_scores
        
        # Filter: High Rating (>= 4.0), not the same course, top similarity
        discovery_results = (
            sim_df[(sim_df['course_name'] != selected_name) & (sim_df['rating'] >= 4.0)]
            .sort_values(by=['similarity', 'rating'], ascending=False)
            .drop_duplicates('course_name')
            .head(5)
        )
        
        st.write(f"**Similar high-rated courses related to '{selected_name}':**")
        st.table(discovery_results[['course_name', 'instructor', 'rating', 'difficulty_level', 'course_price']])
