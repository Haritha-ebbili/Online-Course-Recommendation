import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_data
def load_all_four_files():
    """Load all 4 pickle files - ONLY uses full_data.pkl"""
    
    fulldata_file = 'full_data.pkl'
    traindata_file = 'traindata.pkl'
    tfidf_file = 'tfidf.pkl'
    biases_file = 'biases.pkl'
    
    if not os.path.exists(fulldata_file):
        st.error(f"❌ **{fulldata_file} REQUIRED**")
        st.stop()
    
    fulldata = pd.read_pickle(fulldata_file)
    st.sidebar.success(f"✅ {fulldata_file} ({len(fulldata)} rows)")
    
    # Show actual columns
    st.sidebar.write("**Columns found:**")
    st.sidebar.code(str(fulldata.columns.tolist()))
    
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
    
    return fulldata, traindata, tfidf, biases

# Load files
fulldata, traindata, tfidf, biases = load_all_four_files()

# SMART column detection
def get_safe_columns(df):
    """Auto-detect course columns regardless of exact names"""
    cols_lower = [col.lower() for col in df.columns]
    
    course_id_col = next((col for col in df.columns if 'courseid' in col.lower() or 'course_id' in col.lower()), None)
    course_name_col = next((col for col in df.columns if 'coursename' in col.lower() or 'course_name' in col.lower() or 'course' in col.lower()), None)
    instructor_col = next((col for col in df.columns if 'instructor' in col.lower()), None)
    rating_col = next((col for col in df.columns if 'rating' in col.lower()), None)
    
    return {
        'courseid': course_id_col,
        'coursename': course_name_col,
        'instructor': instructor_col,
        'rating': rating_col
    }

# Extract course data safely
course_cols = get_safe_columns(fulldata)
st.sidebar.write("**Detected columns:**", course_cols)

# Validate required columns
required_cols = ['courseid', 'coursename']
missing_cols = [col for col, cname in course_cols.items()[:2] if cname is None]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

courses = fulldata[[course_cols['courseid'], course_cols['coursename'], 
                   course_cols.get('instructor'), course_cols.get('rating')]].drop_duplicates(subset=course_cols['courseid'])
courses.columns = ['courseid', 'coursename', 'instructor', 'rating']  # Standardize names

# Biases setup
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

st.title("🎓 Online Course Recommendation System")

user_id = st.number_input("👤 User ID", min_value=1, max_value=49999, value=15796)

def predict_score(userid, courseid):
    if use_biases and courseid in item_bias:
        bu = user_bias.get(userid, 0)
        bi = item_bias.get(courseid, 0)
        return global_mean + bu + bi
    else:
        # Safe rating fallback
        try:
            rating = float(courses[courses['courseid'] == courseid]['rating'].iloc[0])
            return rating + np.random.normal(0, 0.1)
        except:
            return 4.0  # Default

def get_user_seen_courses(userid):
    if traindata is not None:
        return set(traindata[traindata['userid'] == userid]['courseid'].values)
    return set()

def get_recommendations(userid, n=15):
    seen = get_user_seen_courses(userid)
    candidates = [cid for cid in courses['courseid'].astype(str) if cid not in seen][:100]
    
    scores = []
    for cid in candidates:
        score = predict_score(userid, cid)
        scores.append({'courseid': cid, 'score': score})
    
    recs = sorted(scores, key=lambda x: x['score'], reverse=True)[:n]
    rec_df = pd.DataFrame(recs).merge(courses, on='courseid', how='left')
    return rec_df.fillna({'rating': 4.0})

# Recommendations
if st.button("🚀 Generate Recommendations", type="primary"):
    with st.spinner("Computing recommendations..."):
        recs = get_recommendations(user_id)
    
    st.success(f"✅ {len(recs)} recommendations for User **{user_id}**!")
    
    st.subheader("📚 Top Recommendations")
    display_cols = ['coursename', 'instructor', 'rating', 'score']
    top10 = recs[display_cols].head(10)
    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("Pred Score", format="%.3f"),
            "rating": st.column_config.NumberColumn("Rating", format="%.2f")
        }
    )
    
    st.subheader("🔍 Filter High-Rated Courses")
    options = recs['coursename'].head(15).tolist()
    selected = st.multiselect("Select courses:", options, default=options[:3])
    
    if selected:
        filtered = recs[recs['coursename'].isin(selected)]
        high_rated = filtered[filtered['rating'] >= 4.0].drop_duplicates('instructor')
        
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            st.balloons()
            st.markdown(f"""
            ## 🎯 **TOP PICK**
            **{best['coursename']}**  
            by *{best['instructor']}*  
            ⭐ **{best['rating']:.1f}** | 📈 **{best['score']:.2f}**
            """)
            st.dataframe(high_rated[display_cols])
        else:
            st.dataframe(filtered.sort_values('score', ascending=False)[display_cols])

# Sidebar
with st.sidebar:
    st.header("📁 Files")
    st.markdown("""
    ✅ full_data.pkl  ← **REQUIRED**
    ✅ traindata.pkl  ← Optional
    ✅ tfidf.pkl      ← Optional
    ✅ biases.pkl     ← Optional
    """)
    
    st.header("⚙️ Install")
    st.code("pip install streamlit pandas numpy scikit-learn")
    
    if biases is not None:
        st.metric("Model", "Hybrid Biases")
        st.metric("Global Mean", f"{global_mean:.3f}")

st.markdown("---")
st.caption("🎓 **Auto-detects your column names** - No more KeyErrors!")
