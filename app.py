import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Course Recommender", layout="wide")

# 🎨 PERFECT COURSE PLATFORM STYLING
st.markdown("""
<style>
    /* Main Title */
    .main-header {
        font-size: 3.5rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Blue Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #1e88e5, #42a5f5) !important;
    }
    
    /* Custom Button - Teal/Education Blue */
    .stButton > button {
        background: linear-gradient(45deg, #26c6da, #00acc1) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 40px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        box-shadow: 0 8px 25px rgba(38, 198, 218, 0.4) !important;
        transition: all 0.4s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(38, 198, 218, 0.6) !important;
        background: linear-gradient(45deg, #00bcd4, #0097a7) !important;
    }
    
    /* Multiselect - Purple/Gold */
    .stMultiSelect > div > div > div {
        border: 3px solid #ab47bc !important;
        border-radius: 15px !important;
        background: linear-gradient(135deg, #f3e5f5, #e1bee7) !important;
    }
    
    /* Dataframe styling */
    .stDataFrame table {
        border-radius: 15px !important;
        overflow: hidden !important;
    }
    .stDataFrame thead tr th {
        background: linear-gradient(90deg, #42a5f5, #1e88e5) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Metrics */
    .stMetric > div {
        background: linear-gradient(135deg, #ff7043, #ff5722) !important;
        border-radius: 20px !important;
        padding: 20px !important;
    }
    
    /* Headers */
    .stMarkdown h2 {
        color: #1e88e5 !important;
        border-bottom: 3px solid #42a5f5 !important;
        padding-bottom: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_pickle('full_data.pkl')

df = load_data()

st.markdown('<h1 class="main-header">🎓 Course Recommendation System</h1>', unsafe_allow_html=True)

st.header("Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

st.header("Step 2: Number of Recommendations")
num_recommendations = st.slider("How many unique courses?", 1, 20, 10, key="blue_slider")

if st.button("🚀 Generate Recommendations", key="generate"):
    # ✅ UNIQUE COURSE NAMES ONLY
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
        hide_index=True,
        column_config={
            "Pred Score": st.column_config.NumberColumn(format="%.2f"),
            "Rating": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    # Store UNIQUE course names
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
        
        # Step 5: DIRECT DATAFRAME (4.0-5.0) - NO PREVIOUS COLUMNS
        high_rated = selected_df[
            (selected_df['Rating'] >= 4.0) & (selected_df['Rating'] <= 5.0)
        ]
        
        st.header("Step 5: Selected Courses (4.0-5.0)")
        if len(high_rated) > 0:
            # DIRECT DATAFRAME - NO METRICS FIRST
            st.dataframe(high_rated[['Course Name', 'Instructor', 'Rating', 'Pred Score']])
            
            # Metrics AFTER dataframe
            best = high_rated.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Top Course", best['Course Name'][:40])
            col2.metric("⭐ Rating", f"{best['Rating']:.2f}")
            col3.metric("📈 Score", f"{best['Pred Score']:.2f}")
        else:
            st.dataframe(selected_df[['Course Name', 'Instructor', 'Rating', 'Pred Score']])
