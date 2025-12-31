import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Course Recommender", layout="wide")

# --- LOAD DATA ---
@st.cache_resource
def load_all():
    try:
        # Check file sizes to prevent 'Ran out of input'
        for f_name in ['full_data.pkl', 'train_data.pkl', 'biases.pkl', 'tfidf.pkl']:
            if os.path.getsize(f_name) == 0:
                st.error(f"File {f_name} is empty. Please re-generate it.")
                return None
        
        full_df = pd.read_pickle('full_data.pkl')
        train_df = pd.read_pickle('train_data.pkl')
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        
        # Prepare lookup and TF-IDF matrix
        lookup = full_df[['course_id', 'course_name', 'instructor', 'rating']].drop_duplicates('course_id')
        features = full_df['course_name'].astype(str) + ' ' + full_df['instructor'].astype(str)
        tfidf_matrix = tfidf.transform(features)
        
        return full_df, train_df, biases, tfidf_matrix, lookup
    except Exception as e:
        st.error(f"Error: {e}")
        return None

data = load_all()

if data:
    df, train, biases, tfidf_matrix, lookup = data
    
    st.title("🎓 Course Recommendation System")
    user_id = st.number_input("Enter User ID:", value=15796)

    if st.button("Get Recommendations"):
        # Hybrid Logic (50% CF + 50% Mean)
        g_mean = biases['global_mean']
        u_b = biases['user_bias']
        i_b = biases['item_bias']
        
        seen = train[train['userid'] == user_id]['courseid'].tolist()
        candidates = [c for c in train['courseid'].unique() if c not in seen]
        
        results = []
        for cid in candidates:
            cf_score = g_mean + u_b.get(user_id, 0) + i_b.get(cid, 0)
            item_mean = train[train['courseid'] == cid]['rating'].mean()
            score = 0.5 * cf_score + 0.5 * item_mean
            results.append((cid, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top_5 = pd.DataFrame(results[:5], columns=['course_id', 'recommendation_score'])
        
        # Merge with metadata
        final_recs = top_5.merge(lookup, on='course_id', how='left')
        
        # Exact formatting
        final_recs['recommendation_score'] = final_recs['recommendation_score'].map('{:.6f}'.format)
        
        st.subheader(f"Top Recommendations for User {user_id}")
        st.table(final_recs[['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']])
        st.session_state['recs'] = final_recs

    # --- SIMILAR COURSE DISCOVERY ---
    if 'recs' in st.session_state:
        st.divider()
        st.subheader("🔍 Explore Best Rated Similar Courses")
        selected_course = st.selectbox("Select a course to see related high-rated options:", 
                                     ["Select..."] + st.session_state['recs']['course_name'].tolist())
        
        if selected_course != "Select...":
            idx = df[df['course_name'] == selected_course].index[0]
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            temp_df = df.copy()
            temp_df['sim'] = sim_scores
            # Filter: High similarity, high rating, and not the same course
            similar = temp_df[(temp_df['course_name'] != selected_course) & (temp_df['rating'] >= 4.5)]
            similar = similar.sort_values(by=['sim', 'rating'], ascending=False).head(5)
            
            st.write(f"Best matches for **{selected_course}**:")
            st.table(similar[['course_name', 'instructor', 'rating', 'course_price']])
