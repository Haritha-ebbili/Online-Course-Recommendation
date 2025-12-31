import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Course Recommender", layout="wide")

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #1f77b4 !important;
        font-weight: bold !important;
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #FF5252, #26A69A);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_pickle('full_data.pkl')

df = load_data()

st.markdown('<h1 class="main-header">Course Recommendation System</h1>', unsafe_allow_html=True)

st.header("Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

st.header("Step 2: Number of Recommendations")
num_recommendations = st.slider("How many unique courses?", 1, 20, 10)

# Advanced: Custom high-rating threshold (3.5 to 5)
st.header("Step 3: High-Rating Threshold")
rating_threshold = st.slider("Minimum rating for 'High-Rated' courses:", 3.5, 5.0, 4.0, 0.1)

if st.button("🚀 Generate Recommendations", key="generate", help="Click to get personalized recommendations"):
    unique_courses = df.drop_duplicates(subset='course_id')
    
    unique_courses['rating'] = unique_courses['rating']
    course_col = 'course_name'
    instructor_col = 'instructor'
    
    unique_courses['score'] = unique_courses['rating'] + np.random.normal(0, 0.1, len(unique_courses))
    recommendations = unique_courses.nlargest(num_recommendations, 'score')
    
    st.header("Step 4: Recommended Courses")
    display_cols = [course_col, instructor_col, 'rating', 'score']
    rec_display = recommendations[display_cols].round(2)
    rec_display.columns = ['Course Name', 'Instructor', 'Rating', 'Pred Score']
    
    st.dataframe(
        rec_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pred Score": st.column_config.NumberColumn(format="%.2f"),
            "Rating": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    st.session_state.recommendations = rec_display
    st.session_state.course_options = rec_display['Course Name'].tolist()
    st.session_state.rating_threshold = rating_threshold

if 'recommendations' in st.session_state:
    st.header("Step 5: Select Courses")
    selected_courses = st.multiselect(
        "Choose courses:",
        st.session_state.course_options,
        default=[]
    )
    
    if selected_courses:
        selected_df = st.session_state.recommendations[
            st.session_state.recommendations['Course Name'].isin(selected_courses)
        ]
        
        threshold = st.session_state.rating_threshold
        high_rated = selected_df[selected_df['Rating'] >= threshold]
        
        st.header(f"Step 6: High-Rated Selected Courses (≥{threshold})")
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Top Course", best['Course Name'][:40])
            with col2:
                st.metric("⭐ Rating", f"{best['Rating']:.2f}")
            with col3:
                st.metric("📈 Score", f"{best['Pred Score']:.2f}")
            
            st.dataframe(high_rated[['Course Name', 'Instructor', 'Rating', 'Pred Score']])
        else:
            st.info(f"No courses ≥{threshold} rating selected")
            st.dataframe(selected_df[['Course Name', 'Instructor', 'Rating', 'Pred Score']])
