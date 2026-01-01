import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Course Recommender", layout="wide")
st.markdown("""
<style>
    /* FULL PAGE BACKGROUND COLOR */
    .stApp {
        background-color: #D5CABD !important;
    }
    /* NEW HEADING COLOR - GOLD */
    .main-header
    section[data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%) !important;
    }
    .main-header {
    font-size: 3.5rem !important;
    background: linear-gradient(90deg, #6a1b9a, #ec407a) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    }

   .stSlider > div > div > div > div {
    background-color: #7b1fa2 !important;
    height: 10px !important;
    border-radius: 12px !important;
    }


    .stButton > button {
    background: linear-gradient(45deg, #1e3c72, #7b1fa2) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 15px 40px !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    box-shadow: 0 8px 25px rgba(123, 31, 162, 0.45) !important;
    transition: all 0.35s ease !important;
    }

    .stButton > button:hover {
        background: #3596B5 !important;
        transform: translateY(-3px) scale(1.02) !important;
    }

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

if st.button(" Generate Recommendations"):
    #  UNIQUE COURSE NAMES ONLY
    unique_by_name = df.drop_duplicates(subset='course_name')
    
    unique_by_name['score'] = unique_by_name['rating'] + np.random.normal(0, 0.1, len(unique_by_name))
    recommendations = unique_by_name.nlargest(num_recommendations, 'score')
    
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
    st.session_state.course_options = rec_display['Course Name'].tolist()

if 'recommendations' in st.session_state:
    st.header("Step 4: Select Courses")
    selected_courses = st.multiselect(
        "Choose courses:",
        st.session_state.course_options,
        default=[]
    )
    
    if selected_courses:
        #  Only ratings between 4 and 5
        step5_result = df[
            (df['course_name'].isin(selected_courses)) &
            (df['rating'] >= 4) &
            (df['rating'] <= 5)
        ][['course_name', 'instructor', 'rating']].drop_duplicates()

        step5_result = step5_result.sort_values(
            by=['course_name', 'rating'], ascending=[True, False]
        )

        step5_result.columns = ['Course Name', 'Instructor', 'Rating']

        st.header("Step 5: Same Course – Different Instructors (Rating 4–5 Only)")
        st.dataframe(
            step5_result,
            use_container_width=True,
            hide_index=True
        )
