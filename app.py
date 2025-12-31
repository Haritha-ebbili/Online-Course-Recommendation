import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_resource
def load_assets():
    files = ['full_data.pkl', 'train_data.pkl', 'biases.pkl', 'tfidf.pkl']
    if not all(os.path.exists(f) for f in files):
        return None
    
    try:
        full_df = pd.read_pickle('full_data.pkl')
        train_df = pd.read_pickle('train_data.pkl')
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
            
        # Remove duplicates from lookup to prevent duplicate display rows
        lookup = full_df[['course_id', 'course_name', 'instructor', 'rating']].drop_duplicates(subset=['course_id'])
        
        # Pre-process TF-IDF
        full_df['features'] = full_df['course_name'].astype(str) + ' ' + full_df['instructor'].astype(str)
        tfidf_matrix = tfidf.transform(full_df['features'])
        
        return full_df, train_df, biases, tfidf_matrix, lookup
    except:
        return None

assets = load_assets()

if assets:
    df, train, biases, tfidf_matrix, lookup = assets
    st.title("🎓 Online Course Recommendation System")
    
    user_id = st.number_input("Enter User ID:", value=15796)

    if st.button("Generate Recommendations"):
        g_mean = biases['global_mean']
        u_b = biases['user_bias']
        i_b = biases['item_bias']
        
        # Hybrid Scoring
        seen = train[train['userid'] == user_id]['courseid'].tolist()
        candidates = [c for c in train['courseid'].unique() if c not in seen]
        
        results = []
        for cid in candidates:
            cf_score = g_mean + u_b.get(user_id, 0) + i_b.get(cid, 0)
            item_mean = train[train['courseid'] == cid]['rating'].mean()
            # 50% Collaborative + 50% Content/Mean
            score = 0.5 * cf_score + 0.5 * item_mean
            results.append((cid, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top_df = pd.DataFrame(results[:5], columns=['course_id', 'recommendation_score'])
        
        # Merge and format
        final = top_df.merge(lookup, on='course_id', how='left')
        final['recommendation_score'] = final['recommendation_score'].map('{:.6f}'.format)
        
        st.subheader(f"Top Recommendations for User {user_id}")
        st.table(final[['course_id', 'recommendation_score', 'course_name', 'instructor', 'rating']])
        st.session_state['recs'] = final

    # FEATURE: Select course for further recommendation
    if 'recs' in st.session_state:
        st.divider()
        st.subheader("🎯 Best Similar Courses")
        
        # We use Instructor name in dropdown to distinguish duplicate course names
        choice = st.selectbox("Select a course to find similar high-rated options:", 
                              ["Select..."] + (st.session_state['recs']['course_name'] + " by " + st.session_state['recs']['instructor']).tolist())
        
        if choice != "Select...":
            # Extract just the course name from the selection
            selected_course_name = choice.split(" by ")[0]
            idx = df[df['course_name'] == selected_course_name].index[0]
            
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            temp_df = df.copy()
            temp_df['sim'] = sim_scores
            
            # Filter for High Rating and High Similarity
            similar = temp_df[(temp_df['course_name'] != selected_course_name) & (temp_df['rating'] >= 4.0)]
            similar = similar.sort_values(by=['sim', 'rating'], ascending=False).drop_duplicates('course_name').head(5)
            
            st.table(similar[['course_name', 'instructor', 'rating', 'course_price']])
else:
    st.error("System files missing. 1. Run your notebook cells. 2. Download the .pkl files. 3. Place them in the same folder as app.py.")
