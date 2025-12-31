import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_data
def load_data():
    """Load full_data.pkl (REQUIRED)"""
    if not os.path.exists('full_data.pkl'):
        st.error("❌ **full_data.pkl REQUIRED**")
        st.stop()
    
    data = pd.read_pickle('full_data.pkl')
    st.sidebar.success(f"✅ full_data.pkl loaded ({len(data)} rows)")
    return data

# Load data
df = load_data()

# Smart column detection
def find_columns(df):
    cols_lower = [col.lower() for col in df.columns]
    return {
        'courseid': next((col for col in df.columns if any(x in col.lower() for x in ['courseid', 'course_id'])), df.columns[1]),
        'coursename': next((col for col in df.columns if any(x in col.lower() for x in ['coursename', 'course_name', 'course'])), df.columns[2]),
        'instructor': next((col for col in df.columns if 'instructor' in col.lower()), None),
        'rating': next((col for col in df.columns if 'rating' in col.lower()), None)
    }

cols = find_columns(df)
courses = df[[cols['courseid'], cols['coursename'], cols.get('instructor', ''), cols.get('rating', 4.0)]].drop_duplicates(subset=cols['courseid'])
courses.columns = ['courseid', 'coursename', 'instructor', 'rating']

st.title("🎓 Course Recommendation System")

# STEP 1: User ID Input
st.header("👤 Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

# STEP 2: Number of recommendations (SLIDER in sidebar)
st.header("📊 Step 2: Select Number of Recommendations")
num_recommendations = st.sidebar.slider("How many courses to recommend?", 5, 20, 10)

# Generate recommendations button
if st.button("🚀 Generate Recommendations", type="primary"):
    # Simple recommendation logic (highest ratings)
    courses['pred_score'] = courses['rating'] + np.random.normal(0, 0.1, len(courses))
    recommendations = courses.nlargest(num_recommendations, 'pred_score')
    
    st.success(f"✅ Generated {len(recommendations)} recommendations for User **{user_id}**!")
    
    # STEP 3: Show recommendations
    st.header("📚 Step 3: Recommended Courses")
    st.dataframe(
        recommendations[['coursename', 'instructor', 'rating', 'pred_score']].round(2),
        use_container_width=True,
        hide_index=True,
        column_config={
            "pred_score": st.column_config.NumberColumn("Score", format="%.2f"),
            "rating": st.column_config.NumberColumn("Rating", format="%.2f")
        }
    )
    
    # STEP 4: Select courses from recommendations
    st.header("🔍 Step 4: Select Courses")
    course_options = recommendations['coursename'].tolist()
    selected_courses = st.multiselect(
        "Choose courses from recommendations:",
        course_options,
        default=course_options[:3]
    )
    
    # STEP 5: Show high-rated selected courses
    if selected_courses:
        selected_df = recommendations[recommendations['coursename'].isin(selected_courses)]
        high_rated = selected_df[selected_df['rating'] >= 4.0].drop_duplicates(subset='instructor')
        
        if len(high_rated) > 0:
            st.balloons()
            st.header("🎯 **HIGH-RATED SELECTED COURSES (4.0+)**")
            
            # Best pick
            best = high_rated.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Top Pick", best['coursename'][:30] + "...")
            col2.metric("⭐ Rating", f"{best['rating']:.2f}")
            col3.metric("👨‍🏫 Instructor", best['instructor'])
            
            # All high-rated options
            st.dataframe(high_rated[['coursename', 'instructor', 'rating', 'pred_score']].round(2))
        else:
            st.warning("⚠️ No 4.0+ rated courses in selection. Showing all:")
            st.dataframe(selected_df[['coursename', 'instructor', 'rating', 'pred_score']].round(2))

# Sidebar instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Enter User ID**
    2. **Use slider** → Choose # of recommendations (5-20)
    3. **Click Generate** → See recommendations
    4. **Select courses** → Get high-rated (4.0+) picks
    """)
    
    st.header("📁 Files")
    st.markdown("✅ **full_data.pkl** ← REQUIRED")
    
    st.header("⚙️ Setup")
    st.code("""
pip install streamlit pandas numpy
streamlit run app.py
    """)

st.markdown("---")
st.caption("🎓 **Exact flow**: User ID → Slider → Recommendations → Select → High-rated courses")
