import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_data
def load_all_four_files():
    """Load all 4 pickle files - ONLY uses full_data.pkl"""
    
    # REQUIRED: full_data.pkl only
    fulldata_file = 'full_data.pkl'
    traindata_file = 'traindata.pkl'
    tfidf_file = 'tfidf.pkl'
    biases_file = 'biases.pkl'
    
    # Load fulldata (REQUIRED)
    if not os.path.exists(fulldata_file):
        st.error(f"❌ **{fulldata_file} REQUIRED**")
        st.stop()
    
    fulldata = pd.read_pickle(fulldata_file)
    st.sidebar.success(f"✅ {fulldata_file}")
    
    # Load other files (optional)
    traindata = pd.read_pickle(traindata_file) if os.path.exists(traindata_file) else None
    tfidf = None
    biases = None
    
    if os.path.exists(tfidf_file):
        with open(tfidf_file, 'rb') as f:
            tfidf = pickle.load(f)
        st.sidebar.success(f"✅ {tfidf_file}")
    
    if os.path.exists(biases_file):
        with open(biases_file, 'rb') as f:
            biases = pickle.load(f)
        st.sidebar.success(f"✅ {biases_file}")
    
    status_msg = f"✅ full_data.pkl ({len(fulldata)} rows)"
    if traindata is not None: status_msg += f" | traindata ({len(traindata)} rows)"
    if tfidf is not None: status_msg += " | tfidf"
    if biases is not None: status_msg += " | biases"
    
    st.sidebar.success(status_msg)
    return fulldata, traindata, tfidf, biases

# Load ALL 4 files
fulldata, traindata, tfidf, biases = load_all_four_files()

# Extract biases if available
if biases is not None:
    global_mean = biases.get('globalmean', 4.0)
    user_bias = biases.get('userbias', {})
    item_bias = biases.get('itembias', {})
    use_biases = True
else:
    global_mean = 4.0
    user_bias = {}
    item_bias = {}
    use_biases = False

# Course catalog
courses = fulldata[['courseid', 'coursename', 'instructor', 'rating']].drop_duplicates(subset='courseid')

st.title("🎓 Online Course Recommendation System")
st.markdown("**Enter User ID** → **Recommendations** → **Select** → **High-rated + Different Instructors**")

# User input
user_id = st.number_input("👤 User ID", min_value=1, max_value=49999, value=15796)

def predict_score(userid, courseid):
    """Hybrid prediction using biases if available"""
    if use_biases:
        bu = user_bias.get(userid, 0)
        bi = item_bias.get(courseid, 0)
        return global_mean + bu + bi
    else:
        # Fallback: rating-based
        course_rating = courses[courses['courseid'] == courseid]['rating'].iloc[0]
        return course_rating + np.random.normal(0, 0.1)

def get_user_seen_courses(userid):
    """Get courses user has already taken"""
    if traindata is not None:
        return set(traindata[traindata['userid'] == userid]['courseid'].values)
    return set()

def get_recommendations(userid, n=15):
    """Generate recommendations"""
    seen = get_user_seen_courses(userid)
    candidates = [cid for cid in courses['courseid'] if cid not in seen][:100]
    
    scores = []
    for cid in candidates:
        score = predict_score(userid, cid)
        scores.append({'courseid': cid, 'score': score})
    
    recs = sorted(scores, key=lambda x: x['score'], reverse=True)[:n]
    rec_df = pd.DataFrame(recs).merge(courses, on='courseid')
    return rec_df

# Generate recommendations
if st.button("🚀 Generate Recommendations", type="primary"):
    with st.spinner("Computing personalized recommendations..."):
        recs = get_recommendations(user_id)
    
    st.success(f"✅ {len(recs)} recommendations for User **{user_id}**!")
    
    # Display top recommendations
    st.subheader("📚 Top 10 Recommendations")
    top10 = recs.head(10)[['coursename', 'instructor', 'rating', 'score']]
    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("Pred Score", format="%.3f"),
            "rating": st.column_config.NumberColumn("Avg Rating", format="%.2f")
        }
    )
    
    # Course selection for filtering
    st.subheader("🔍 Select Courses to Filter")
    options = recs.head(15)['coursename'].tolist()
    selected = st.multiselect(
        "Choose from recommendations:", 
        options, 
        default=options[:3]
    )
    
    if selected:
        filtered = recs[recs['coursename'].isin(selected)]
        
        # High rating (4.5+) + different instructors
        high_rated = filtered[filtered['rating'] >= 4.5].drop_duplicates(subset='instructor')
        
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            st.balloons()
            st.markdown(f"""
            ## 🎯 **PERFECT MATCH**
            **{best['coursename']}**  
            👨‍🏫 *by {best['instructor']}*  
            ⭐ **{best['rating']:.2f}** | 📈 **{best['score']:.3f}**
            """)
            
            st.subheader("⭐ All High-Rated Options (4.5+) w/ Different Instructors")
            st.dataframe(high_rated[['coursename', 'instructor', 'rating', 'score']])
        else:
            st.warning("⚠️ No 4.5+ rated courses. Showing best matches:")
            st.dataframe(filtered.sort_values('score', ascending=False)[['coursename', 'instructor', 'rating', 'score']])

# Sidebar with file status - FIXED SYNTAX
with st.sidebar:
    st.header("📁 REQUIRED FILES")
    st.markdown("""
✅ full_data.pkl     ← REQUIRED
✅ traindata.pkl     ← Optional  
✅ tfidf.pkl         ← Optional
✅ biases.pkl        ← Optional
    """)
    
    st.header("⚙️ Setup")
    st.code("""
pip install streamlit pandas numpy scikit-learn
streamlit run app.py
    """)
    
    if biases is not None:
        st.metric("Model", "Hybrid (Biases)")
        st.metric("Global Mean", f"{global_mean:.3f}")
    else:
        st.info("📊 Using rating-based fallback")

st.markdown("---")
st.caption("🎓 **Uses ONLY full_data.pkl as primary source**")
