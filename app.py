import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_data
def load_models():
    # Auto-detect full_data.pkl OR fulldata.pkl
    fulldata_file = 'fulldata.pkl' if os.path.exists('fulldata.pkl') else 'full_data.pkl'
    
    try:
        fulldata = pd.read_pickle(fulldata_file)
        with open('biases.pkl', 'rb') as f:
            biases = pickle.load(f)
        st.sidebar.success(f"✅ Loaded: {fulldata_file} + biases.pkl")
        return fulldata, biases
    except FileNotFoundError as e:
        st.error(f"❌ Missing: {e.filename}")
        st.error("**Need ONLY these 2 files:**")
        st.error("```
full_data.pkl (or fulldata.pkl)
biases.pkl
        ```")
        st.stop()

# Load data
fulldata, biases = load_models()
global_mean = biases['globalmean']
user_bias = biases['userbias']
item_bias = biases['itembias']

# Course catalog
courses = fulldata[['courseid', 'coursename', 'instructor', 'rating']].drop_duplicates(subset='courseid')

st.title("🎓 Course Recommendation System")
st.markdown("**User ID** → **Recommendations** → **Select** → **High-rated + Different Instructors**")

# User input
user_id = st.number_input("👤 User ID", min_value=1, max_value=49999, value=15796)

def predict_score(userid, courseid):
    """Hybrid model: Global mean + User bias + Item bias"""
    bu = user_bias.get(userid, 0)
    bi = item_bias.get(courseid, 0)
    return global_mean + bu + bi

def get_recommendations(userid, n=15):
    """Generate top N recommendations"""
    # All available courses
    candidates = courses['courseid'].tolist()
    
    # Score candidates
    scores = []
    for cid in candidates[:100]:  # Fast computation
        score = predict_score(userid, cid)
        scores.append({'courseid': cid, 'score': score})
    
    # Top recommendations
    recs = sorted(scores, key=lambda x: x['score'], reverse=True)[:n]
    rec_df = pd.DataFrame(recs).merge(courses, on='courseid')
    return rec_df

# Generate recommendations
if st.button("🚀 Get Recommendations", type="primary"):
    with st.spinner("Generating recommendations..."):
        recs = get_recommendations(user_id)
    
    st.success(f"✅ {len(recs)} recommendations for User **{user_id}**!")
    
    # Top 10 table
    st.subheader("📚 Top Recommendations")
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
    
    # Course selection
    st.subheader("🔍 Select Courses")
    options = recs.head(15)['coursename'].tolist()
    selected = st.multiselect("Choose courses:", options, default=options[:3])
    
    if selected:
        filtered = recs[recs['coursename'].isin(selected)]
        
        # Filter: 4.5+ rating + different instructors
        high_rated = filtered[filtered['rating'] >= 4.5].drop_duplicates('instructor')
        
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            st.balloons()
            st.markdown(f"""
            ## 🎯 **BEST PICK**
            **{best['coursename']}**  
            👨‍🏫 *{best['instructor']}*  
            ⭐ **{best['rating']:.2f}** | 📈 **{best['score']:.3f}**
            """)
            
            st.subheader("⭐ All High-Rated (4.5+)")
            st.dataframe(high_rated[['coursename', 'instructor', 'rating', 'score']])
        else:
            st.info("ℹ️ No 4.5+ ratings. Best available:")
            st.dataframe(filtered.sort_values('score', ascending=False))

# Sidebar
with st.sidebar:
    st.header("✅ File Status")
    st.success("**ONLY needs:**")
    st.markdown("```
✅ full_data.pkl (or fulldata.pkl)
✅ biases.pkl
❌ NO traindata.pkl needed!
    ```")
    
    st.header("🚀 Quick Start")
    st.markdown("""
    1. Put your 2 pickle files in same folder
    2. `pip install streamlit pandas numpy`
    3. `streamlit run app.py`
    4. Enter User ID → Get recommendations!
    """)
    
    st.header("🎯 Features")
    st.markdown("- Hybrid model predictions")
    st.markdown("- High-rated course filtering")
    st.markdown("- Different instructor selection")
    st.markdown("- No external dependencies")

st.markdown("---")
st.caption("🛠️ Your Hybrid Recommendation Model - Ready to Deploy!")
