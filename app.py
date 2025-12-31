import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# Custom CSS for table styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTable { font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA AND MODELS ---
@st.cache_resource
def load_all_assets():
    try:
        full_data = pd.read_pickle('full_data.pkl')
        train_data = pd.read_pickle('train_data.pkl')
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        
        # Prepare content similarity matrix
        # Using name + instructor as per training notebook logic
        full_data['features'] = (
            full_data['course_name'].astype(str) + ' ' + 
            full_data['instructor'].astype(str)
        )
        tfidf_matrix = tfidf.transform(full_data['features'])
        
        return full_data, train_data, biases, tfidf_matrix
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None

df, train, biases, tfidf_matrix = load_all_assets()

# --- PREDICTION LOGIC ---
if df is not None:
    global_mean = biases['global_mean']
    user_bias = biases['user_bias']
    item_bias = biases['item_bias']

    def get_recommendations(user_id, n=5):
        # Courses seen by user
        seen = train[train['userid'] == user_id]['courseid'].tolist()
        # All unique courses
        all_courses = train['courseid'].unique()
        candidates = [c for c in all_courses if c not in seen]
        
        scores = []
        for cid in candidates:
            # CF Part
            bu = user_bias.get(user_id, 0)
            bi = item_bias.get(cid, 0)
            cf_score = global_mean + bu + bi
            
            # Content Part (Mean rating of that course)
            course_ratings = train[train['courseid'] == cid]['rating']
            content_score = course_ratings.mean() if not course_ratings.empty else global_mean
            
            # Hybrid (50/50)
            hybrid_score = 0.5 * cf_score + 0.5 * content_score
            scores.append((cid, hybrid_score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    # --- UI: USER RECOMMENDATIONS ---
    st.title("🎓 Course Recommendation System")
    
    input_user = st.number_input("Enter User ID:", value=15796, step=1)
    
    if st.button("Get Recommendations"):
        recs = get_recommendations(input_user)
        
        if recs:
            # Format results for display
            rec_df = pd.DataFrame(recs, columns=['course_id', 'recommendation_score'])
            
            # Merge with metadata
            lookup = df[['course_id', 'course_name', 'instructor', 'rating']].drop_duplicates('course_id')
            final_recs = rec_df.merge(lookup, on='course_id', how='left')
            
            st.subheader(f"Top Recommendations for User {input_user}")
            st.table(final_recs[['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']])
            
            # Store recommendations in session state for selection feature
            st.session_state['current_recs'] = final_recs
        else:
            st.warning("No recommendations found for this user.")

    # --- UI: SIMILAR COURSE EXPLORER ---
    if 'current_recs' in st.session_state:
        st.divider()
        st.subheader("🔍 Explore Further")
        st.write("Select a course from your recommendations to find similar high-rated options:")
        
        course_options = st.session_state['current_recs']['course_name'].tolist()
        selected_course_name = st.selectbox("Pick a course:", ["Select..."] + course_options)
        
        if selected_course_name != "Select...":
            # Get index of selected course in the full dataframe
            selected_idx = df[df['course_name'] == selected_course_name].index[0]
            
            # Compute Cosine Similarity for that course against all others
            cosine_sim = cosine_similarity(tfidf_matrix[selected_idx], tfidf_matrix).flatten()
            
            # Add similarity and rank
            sim_df = df.copy()
            sim_df['similarity_score'] = cosine_sim
            
            # Filter: High Rating (> 4.0), not the same course, high similarity
            # We sort by similarity AND rating
            similar_courses = (
                sim_df[sim_df['course_name'] != selected_course_name]
                .sort_values(by=['similarity_score', 'rating'], ascending=False)
                .drop_duplicates('course_name')
                .head(5)
            )
            
            st.write(f"**Because you liked '{selected_course_name}', you might also enjoy:**")
            st.table(similar_courses[['course_name', 'instructor', 'rating', 'difficulty_level', 'course_price']])

else:
    st.error("Model files not found. Ensure `full_data.pkl`, `train_data.pkl`, `biases.pkl`, and `tfidf.pkl` are in the app folder.")
