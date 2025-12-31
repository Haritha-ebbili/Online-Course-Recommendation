import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Page config
st.set_page_config(page_title="Course Recommender", layout="wide")

# Smart pickle loader (full_data.pkl OR fulldata.pkl)
@st.cache_data
def load_models():
    fulldata_file = 'fulldata.pkl' if os.path.exists('fulldata.pkl') else 'full_data.pkl'
    
    try:
        fulldata = pd.read_pickle(fulldata_file)
        traindata = pd.read_pickle('traindata.pkl')
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
        return fulldata, traindata, tfidf, biases
    except FileNotFoundError as e:
        st.error(f"❌ Missing: {e.filename}")
        st.error("""
**Need these 4 files:**
        """)
        st.stop()

# Load data
fulldata, traindata, tfidf, biases = load_models()
global_mean = biases['globalmean']
user_bias = biases['userbias']
item_bias = biases['itembias']

# Course info
courselookup = fulldata[['courseid', 'coursename', 'instructor', 'rating']].drop_duplicates(subset='courseid')

st.title("🎓 Course Recommendation System")
st.markdown("**Enter User ID** → **Get Recommendations** → **Select Courses** → **High-rated picks**")

# User input
user_id = st.number_input("👤 User ID", min_value=1, max_value=49999, value=15796)

# Hybrid model prediction
def predict_score(userid, courseid):
    bu = user_bias.get(userid, 0)
    bi = item_bias.get(courseid, 0)
    return global_mean + bu + bi

def get_recommendations(userid, n=15):
    # User's seen courses
    seen = set(traindata[traindata['userid'] == userid]['courseid'].values)
    # Unseen courses
    candidates = set(courselookup['courseid']) - seen
    
    # Score all candidates (limit for speed)
    scores = []
    for cid in list(candidates)[:100]:
        score = predict_score(userid, cid)
        scores.append({'courseid': cid, 'score': score})
    
    # Top N recommendations
    recs = sorted(scores, key=lambda x: x['score'], reverse=True)[:n]
    rec_df = pd.DataFrame(recs).merge(courselookup, on='courseid')
    return rec_df

# RECOMMEND button
if st.button("🚀 Generate Recommendations", type="primary"):
    with st.spinner("Computing recommendations..."):
        recs = get_recommendations(user_id)
    
    st.success(f"✅ {len(recs)} courses recommended for User **{user_id}**!")
    
    # Show top 10
    st.subheader("📊 Top Recommendations")
    top10 = recs.head(10)[['coursename', 'instructor', 'rating', 'score']]
    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("Pred Score", format="%.3f"),
            "rating": st.column_config.NumberColumn("Rating", format="%.2f")
        }
    )
    
    # Select courses
    st.subheader("🔍 Pick Courses to Filter")
    course_options = recs.head(15)['coursename'].tolist()
    selected = st.multiselect("Select from recommendations:", course_options, 
                             default=course_options[:3])
    
    if selected:
        filtered = recs[recs['coursename'].isin(selected)]
        
        # High rating (4.5+) + unique instructors
        high_rated = filtered[filtered['rating'] >= 4.5].drop_duplicates('instructor')
        
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            st.balloons()
            st.markdown(f"""
            ## 🎯 **TOP PICK**
            **{best['coursename']}**  
            👨‍🏫 *{best['instructor']}*  
            ⭐ **{best['rating']:.2f}** | 📈 **{best['score']:.3f}**
            """)
            
            st.subheader("⭐ All 4.5+ Rating Options")
            st.dataframe(high_rated[['coursename', 'instructor', 'rating', 'score']])
        else:
            st.warning("No 4.5+ courses. Best options:")
            st.dataframe(filtered.sort_values('score', ascending=False))

# Sidebar
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Enter **User ID** (1-49999)
    2. Click **Generate Recommendations**
    3. Select courses from list
    4. View **4.5+ rated courses** with **different instructors**
    """)
    
    st.header("✅ File Check")
    files = ['full_data.pkl', 'fulldata.pkl', 'traindata.pkl', 'tfidf.pkl', 'biases.pkl']
    for f in files:
        status = "✅" if os.path.exists(f) else "❌"
        st.write(f"{status} {f}")
    
    st.header("💻 Run")
    st.code("pip install streamlit pandas numpy\nstreamlit run app.py")

st.markdown("---")
st.caption("🛠️ Hybrid Model (Collaborative Filtering + Biases) from your notebook")
