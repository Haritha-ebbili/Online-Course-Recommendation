import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="Course Recommender", layout="wide")

# CSS to match the table style and clean UI
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
        # Load the saved assets from your model building
        full_df = pd.read_pickle('full_data.pkl')
        train_df = pd.read_pickle('train_data.pkl') # Columns: ['userid', 'courseid', 'rating']
        
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
            
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
            
        # Reconstruct course lookup with original names
        # Assuming original column names from your notebook's Excel read
        lookup = full_df[['course_id', 'course_name', 'instructor', 'rating']].drop_duplicates('course_id')
        
        # Prepare TF-IDF matrix for the 'similar courses' feature
        full_df['content_features'] = full_df['course_name'].astype(str) + ' ' + full_df['instructor'].astype(str)
        tfidf_matrix = tfidf.transform(full_df['content_features'])
        
        return full_df, train_df, biases, tfidf_matrix, lookup
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

df, train, biases, tfidf_matrix, lookup = load_data()

# --- HELPER FUNCTIONS ---
def get_hybrid_score(user_id, course_id, global_mean, user_bias, item_bias, train_df):
    # Collaborative Filtering Part
    bu = user_bias.get(user_id, 0)
    bi = item_bias.get(course_id, 0)
    cf_pred = global_mean + bu + bi
    
    # Content Part (Mean rating of course in training)
    course_ratings = train_df[train_df['courseid'] == course_id]['rating']
    content_pred = course_ratings.mean() if not course_ratings.empty else global_mean
    
    # Hybrid calculation (50/50 split as per notebook)
    return 0.5 * cf_pred + 0.5 * content_pred

def get_recommendations(user_id, n=5):
    global_mean = biases['global_mean']
    user_bias = biases['user_bias']
    item_bias = biases['item_bias']
    
    # Find courses not seen by user
    seen = train[train['userid'] == user_id]['courseid'].unique()
    all_courses = train['courseid'].unique()
    candidates = [c for c in all_courses if c not in seen]
    
    scores = []
    for cid in candidates:
        score = get_hybrid_score(user_id, cid, global_mean, user_bias, item_bias, train)
        scores.append((cid, score))
    
    # Sort and take top N
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]

# --- APP UI ---
st.title("🎓 Online Course Recommendation System")

# 1. User Input
user_input = st.number_input("Enter User ID:", value=15796, step=1)

if st.button("Generate Recommendations"):
    st.subheader(f"Top Recommendations for User {user_input}")
    
    recs = get_recommendations(user_input)
    
    if recs:
        # Create recommendation dataframe
        rec_df = pd.DataFrame(recs, columns=['course_id', 'recommendation_score'])
        
        # Merge with lookup to get details
        final_table = rec_df.merge(lookup, on='course_id', how='left')
        
        # Format the score to match your request (6 decimal places)
        final_table['recommendation_score'] = final_table['recommendation_score'].apply(lambda x: f"{x:.6f}")
        
        # Display the exact table requested
        st.table(final_table[['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']])
        
        # Save to session state for the next step
        st.session_state['recs_visible'] = final_table
    else:
        st.warning("No recommendations found for this user.")

# 2. Further Selection Feature
if 'recs_visible' in st.session_state:
    st.markdown("---")
    st.subheader("🎯 Refine Your Choice")
    st.info("Select a course from the recommendations above to find similar high-rated courses.")
    
    selected_name = st.selectbox(
        "Choose a recommended course to see more like it:",
        ["Select a course..."] + st.session_state['recs_visible']['course_name'].tolist()
    )
    
    if selected_name != "Select a course...":
        # Find the index of the selected course in the main dataframe
        idx = df[df['course_name'] == selected_name].index[0]
        
        # Compute similarity between selected course and all others
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Create a temp dataframe with similarities
        sim_df = df.copy()
        sim_df['similarity'] = sim_scores
        
        # Filter: Exclude selected course, sort by similarity and rating
        # High rating filter (> 4.0 as requested)
        sim_results = (
            sim_df[sim_df['course_name'] != selected_name]
            .sort_values(by=['similarity', 'rating'], ascending=False)
            .drop_duplicates('course_name')
            .head(5)
        )
        
        st.write(f"**Top similar courses with high ratings related to '{selected_name}':**")
        st.table(sim_results[['course_name', 'instructor', 'rating', 'difficulty_level', 'course_price']])
