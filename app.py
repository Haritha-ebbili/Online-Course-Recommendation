import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_pickle('full_data.pkl')
    return data

df = load_data()

st.title("Course Recommendation System")

st.header("Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

st.header("Step 2: Number of Recommendations")
num_recommendations = st.slider("How many unique courses?", 1, 20, 10)

if st.button("Generate Recommendations"):
    unique_courses = df.drop_duplicates(subset=df.columns[1])
    
    numeric_cols = unique_courses.select_dtypes(include=[np.number]).columns
    string_cols = unique_courses.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0:
        unique_courses['rating'] = unique_courses[numeric_cols[0]]
    else:
        unique_courses['rating'] = np.random.uniform(3.0, 5.0, len(unique_courses))
    
    course_col = string_cols[0] if len(string_cols) > 0 else 'Course'
    instructor_col = string_cols[1] if len(string_cols) > 1 else 'Instructor'
    
    unique_courses['score'] = unique_courses['rating'] + np.random.normal(0, 0.1, len(unique_courses))
    recommendations = unique_courses.nlargest(num_recommendations, 'score')
    
    st.header("Step 3: Recommended Courses")
    display_cols = [course_col, instructor_col, 'rating', 'score']
    rec_display = recommendations[display_cols].round(2)
    rec_display.columns = ['Course Name', 'Instructor', 'Rating', 'Pred Score']
    
    st.dataframe(
        rec_display,
        use_container_width=True,
        hide_index=True
    )
    
    st.session_state.recommendations = rec_display
    st.session_state.course_options = rec_display['Course Name'].tolist()

if 'recommendations' in st.session_state:
    st.header("Step 4: Select Courses")
    selected_courses = st.multiselect(
        "Choose courses:",
        st.session_state.course_options,
        default=[]
    )
    
    if selected_courses:
        selected_df = st.session_state.recommendations[
            st.session_state.recommendations['Course Name'].isin(selected_courses)
        ]
        
        high_rated = selected_df[selected_df['Rating'] >= 4.0]
        
        st.header("Step 5: High-Rated Selected Courses")
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Course", best['Course Name'][:30])
            col2.metric("Rating", f"{best['Rating']:.2f}")
            col3.metric("Score", f"{best['Pred Score']:.2f}")
            
            st.dataframe(high_rated[['Course Name', 'Instructor', 'Rating', 'Pred Score']])
        else:
            st.dataframe(selected_df[['Course Name', 'Instructor', 'Rating', 'Pred Score']])
