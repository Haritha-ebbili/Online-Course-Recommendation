import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Course Recommender", layout="wide")

st.markdown("""
<style>
    section[data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%) !important;
    }
    .main-header {
        font-size: 3.5rem !important;
        background: linear-gradient(135deg, #1976d2, #42a5f5) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ff9800, #ffc107) !important;
    }
    .stButton > button {
        background: linear-gradient(45deg, #2e7d32, #4caf50) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 40px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4) !important;
        transition: all 0.4s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #388e3c, #66bb6a) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(76, 175, 80, 0.6) !important;
    }
    .stMultiSelect > div > div > div {
        border: 3px solid #7b1fa2 !important;
        border-radius: 15px !important;
        background: linear-gradient(135deg, #f3e5f5, #e1bee7) !important;
    }
    .stDataFrame table {
        border-radius: 15px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    }
    .stDataFrame thead tr th {
        background: linear-gradient(90deg, #1976d2, #42a5f5) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    .stMarkdown h2 {
        color: #1976d2 !important;
        border-bottom: 4px solid #42a5f5 !important;
        padding-bottom: 12px !important;
        background: rgba(255,255,255,0.8) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
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

if st.button("🚀 Generate Recommendations"):
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
        
        high_rated = selected_df[
            (selected_df['Rating'] >= 4.0) & (selected_df['Rating'] <= 5.0)
        ].drop_duplicates(subset='Instructor')
        
        st.header("Step 5: Selected Courses from Recommendations (4.0-5.0)")
        st.dataframe(high_rated[['Course Name', 'Instructor', 'Rating', 'Pred Score']])
        
        if len(high_rated) > 0:
            best = high_rated.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Top Course", best['Course Name'][:40])
            col2.metric("⭐ Rating", f"{best['Rating']:.2f}")
            col3.metric("📈 Score", f"{best['Pred Score']:.2f}")
