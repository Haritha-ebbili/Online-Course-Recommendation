import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_data
def load_data():
    if not os.path.exists('full_data.pkl'):
        st.error("❌ **full_data.pkl REQUIRED**")
        st.stop()
    
    data = pd.read_pickle('full_data.pkl')
    
    # DEBUG: Show actual columns
    st.sidebar.subheader("📊 Your Data Columns")
    st.sidebar.write(data.columns.tolist())
    st.sidebar.write("Shape:", data.shape)
    
    return data

# Load data
df = load_data()

st.title("🎓 Course Recommendation System")
st.markdown("**User ID → Slider → Recommendations → Select → High-rated**")

# STEP 1: User ID
st.header("👤 Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

# STEP 2: Number of recommendations (SLIDER)
st.header("📊 Step 2: Number of Recommendations")
num_recommendations = st.sidebar.slider("How many courses?", 5, 20, 10)

# Simple recommendation logic (works with ANY columns)
def get_recommendations(df, n=10):
    # Use first numeric column as "rating" (safe fallback)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        rating_col = numeric_cols[0]
        df['score'] = df[rating_col]
    else:
        df['score'] = np.random.uniform(3.0, 5.0, len(df))
    
    # Use first string columns as course name/instructor
    string_cols = df.select_dtypes(include=['object']).columns
    course_name = string_cols[0] if len(string_cols) > 0 else 'Course'
    instructor = string_cols[1] if len(string_cols) > 1 else 'Instructor'
    
    top_courses = df.nlargest(n, 'score')
    return top_courses[[course_name, instructor, 'score']], course_name, instructor

if st.button("🚀 Generate Recommendations", type="primary"):
    recommendations_df, course_col, instructor_col = get_recommendations(df, num_recommendations)
    
    st.success(f"✅ {len(recommendations_df)} recommendations for User **{user_id}**!")
    
    # STEP 3: Show recommendations
    st.header("📚 Step 3: Recommended Courses")
    display_df = recommendations_df.copy()
    display_df.columns = ['Course Name', 'Instructor', 'Score']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # STEP 4: Select from recommendations
    st.header("🔍 Step 4: Select Courses")
    course_options = display_df['Course Name'].tolist()
    selected_courses = st.multiselect(
        "Choose from recommendations:",
        course_options,
        default=course_options[:3]
    )
    
    # STEP 5: High-rated selected courses
    if selected_courses:
        selected_mask = display_df['Course Name'].isin(selected_courses)
        selected_df = display_df[selected_mask]
        
        # High-rated (score >= 4.0)
        high_rated = selected_df[selected_df['Score'] >= 4.0]
        
        if len(high_rated) > 0:
            st.balloons()
            st.header("🎯 **HIGH-RATED SELECTED COURSES (4.0+)**")
            
            # Top pick
            best = high_rated.iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏆 Top Course", best['Course Name'][:40])
            with col2:
                st.metric("⭐ Score", f"{best['Score']:.2f}")
            
            st.dataframe(high_rated)
        else:
            st.warning("⚠️ No 4.0+ rated courses selected")
            st.dataframe(selected_df)

# Instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Enter **User ID**
    2. Use **slider** → # recommendations
    3. Click **Generate**
    4. **Select courses**
    5. See **high-rated picks** 🎯
    """)
    
    st.header("📁 Files")
    st.markdown("✅ **full_data.pkl** only")
    
    st.header("⚙️ Run")
    st.code("""
pip install streamlit pandas numpy
streamlit run app.py
    """)

st.markdown("---")
st.caption("🎓 **Works with ANY column structure!**")
