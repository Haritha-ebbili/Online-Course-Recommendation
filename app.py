import streamlit as st
import pandas as pd
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Course Recommender", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp { background-color: #D5CABD !important; }
.main-header {
    font-size: 3rem !important;
    background: linear-gradient(90deg, #1e3c72, #7b1fa2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    text-align: center !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(45deg, #1e3c72, #7b1fa2) !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ================= DATA LOADING =================
@st.cache_data
def load_data():
    # Check if file exists, if not create dummy data for demonstration
    if os.path.exists("full_data.pkl"):
        return pd.read_pickle("full_data.pkl")
    else:
        # Dummy data if file is missing
        data = {
            'user_id': np.random.randint(15790, 15800, 100),
            'course_id': np.random.randint(100, 500, 100),
            'course_name': [f"Course {i}" for i in range(100)],
            'instructor': [f"Instructor {i}" for i in range(100)],
            'rating': np.random.uniform(3.5, 5.0, 100)
        }
        return pd.DataFrame(data)

df = load_data()

# ================= INITIALIZE SESSION STATE =================
if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# ================= UI =================
st.markdown('<h1 class="main-header">Course Recommendation System</h1>', unsafe_allow_html=True)

# Step 1 & 2: Inputs
col1, col2 = st.columns(2)
with col1:
    st.header("Step 1: User Profile")
    user_id_input = st.number_input("Enter User ID", min_value=1, value=15796)

with col2:
    st.header("Step 2: Preferences")
    num_recs = st.slider("Quantity", 1, 20, 10)

# Generate Button
if st.button("Generate Recommendations"):
    # 1. Get user history
    user_history = df[df["user_id"] == user_id_input]["course_name"].unique()
    
    # 2. Filter out already taken courses
    potential_courses = df[~df["course_name"].isin(user_history)]
    
    # 3. Ensure uniqueness and calculate score (User-Specific: Based on high ratings in dataset)
    # We group by course_name to ensure 100% unique names
    recommendations = potential_courses.groupby('course_name').agg({
        'course_id': 'first',
        'instructor': 'first',
        'rating': 'mean'
    }).reset_index()
    
    # Sort by top ratings
    recommendations = recommendations.sort_values(by="rating", ascending=False).head(num_recs)
    recommendations["recommendation_score"] = recommendations["rating"].round(2)
    
    # Save to session state
    st.session_state.recommendations = recommendations
    st.session_state.clicked = True

# ================= OUTPUT STEPS =================

# STEP 3: Display Recommendations
if st.session_state.clicked and st.session_state.recommendations is not None:
    st.divider()
    st.header("Step 3: Recommended Courses (User-Specific)")
    
    # Filter columns for display
    display_df = st.session_state.recommendations[["course_id", "recommendation_score", "course_name", "instructor", "rating"]]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # STEP 4: Selection
    st.header("Step 4: Select Courses to Compare")
    course_list = st.session_state.recommendations["course_name"].tolist()
    selected_courses = st.multiselect("Pick from the list above:", options=course_list)

    # STEP 5: Final Result
    if selected_courses:
        st.header("Step 5: Selected Courses (Detailed View)")
        
        # Filter original dataframe for selected courses with high ratings
        # We ensure uniqueness here too
        step5_result = df[df["course_name"].isin(selected_courses)].copy()
        step5_result = step5_result.drop_duplicates(subset="course_name")
        
        # Display table
        st.table(step5_result[["course_id", "course_name", "instructor", "rating"]])
    else:
        st.info("Select courses in Step 4 to see the final details.")
