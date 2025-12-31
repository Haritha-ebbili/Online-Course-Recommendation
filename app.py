import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Course Recommender", layout="wide")

# 🖤 BLACK THEME - Course Recommendation System
st.markdown("""
<style>
    /* Black Theme */
    .main-header {
        font-size: 3.5rem !important;
        color: #000000 !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Black Background */
    section[data-testid="stAppViewContainer"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Black Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #333333, #000000) !important;
    }
    
    /* Black Button */
    .stButton > button {
        background: linear-gradient(45deg, #000000, #333333) !important;
        color: #ffffff !important;
        border: 2px solid #666666 !important;
        border-radius: 50px !important;
        padding: 15px 40px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.8) !important;
        transition: all 0.4s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #333333, #000000) !important;
        box-shadow: 0 12px 35px rgba(0,0,0,1) !important;
        border-color: #ffffff !important;
    }
    
    /* Black Multiselect */
    .stMultiSelect > div > div > div {
        border: 3px solid #444444 !important;
        border-radius: 15px !important;
        background: #2a2a2a !important;
        color: #ffffff !important;
    }
    
    /* Black Dataframe */
    .stDataFrame table {
        border-radius: 15px !important;
        background: #2a2a2a !important;
        color: #ffffff !important;
    }
    .stDataFrame thead tr th {
        background: linear-gradient(90deg, #000000, #333333) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stDataFrame tbody tr td {
        background: #2a2a2a !important;
        color: #ffffff !important;
        border-color: #444444 !important;
    }
    
    /* Black Headers */
    .stMarkdown h2 {
        color: #ffffff !important;
        border-bottom: 3px solid #333333 !important;
        padding-bottom: 10px !important;
        background: #000000 !important;
        padding: 15px !important;
        border-radius: 10px !important;
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

if st.button("Generate Recommendations"):
    unique_courses = df.drop_duplicates(subset=['course_id', 'course_name'])
    
    unique_courses['score'] = unique_courses['rating'] + np.random.normal(0, 0.1, len(unique_courses))
    recommendations = unique_courses.nlargest(num_recommendations, 'score')
    
    st.header("Step 3: Recommended Courses")
    display_cols = ['course_name', 'instructor', 'rating', 'score']
    rec_display = recommendations[display_cols].round(2)
    rec_display.columns = ['Course Name', 'Instructor', 'Rating', 'Pred Score']
    
    st.dataframe(
        rec_display,
        use_container_width=True,
        hide_index=True
    )
    
    st.session_state.recommendations = rec_display
    st.session_state.course_options = rec_display['Course Name'].drop_duplicates().tolist()

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
        
        # ✅ Step 5: 4.0-5.0 + DIFFERENT INSTRUCTORS
        high_rated = selected_df[
            (selected_df['Rating'] >= 4.0) & (selected_df['Rating'] <= 5.0)
        ].drop_duplicates(subset='Instructor')
        
        st.header("Step 5: Selected Courses (4.0-5.0, Different Instructors)")
        st.dataframe(high_rated[['Course Name', 'Instructor', 'Rating', 'Pred Score']])
        
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Course", best['Course Name'][:40])
            col2.metric("Rating", f"{best['Rating']:.2f}")
            col3.metric("Score", f"{best['Pred Score']:.2f}")
