import streamlit as st
import pandas as pd
import numpy as np

# PAGE CONFIG 
st.set_page_config(page_title="Course Recommender", layout="wide")

# CUSTOM CSS (Kept your original styling)
st.markdown("""
<style>
.stApp { background-color: #D5CABD !important; }
.main-header {
    font-size: 3.5rem !important;
    background: linear-gradient(90deg, #6a1b9a, #ec407a) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
}
/* BUTTON */
.stButton > button {
    background: linear-gradient(45deg, #6a1b9a, #4527a0) !important; /* Deep Purple Gradient */
    color: white !important;
    border: none !important;
    border-radius: 12px !important; /* Slightly more modern than 50px */
    padding: 12px 30px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(106, 27, 154, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100%; /* Makes the button full width for better UX */
}

.stButton > button:hover {
    background: linear-gradient(45deg, #4527a0, #6a1b9a) !important; /* Reverse gradient on hover */
    box-shadow: 0 6px 20px rgba(106, 27, 154, 0.4) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}
</style>
""", unsafe_allow_html=True)

# LOAD DATA
@st.cache_data
def load_data():
    # Ensure your pkl file contains 'course_id'
    return pd.read_pickle("full_data.pkl")

df = load_data()

# TITLE 
st.markdown('<h1 class="main-header">Course Recommendation System</h1>', unsafe_allow_html=True)

# STEP 1 
st.header("Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

# STEP 2 
st.header("Step 2: Number of Recommendations")
num_recommendations = st.slider("How many unique courses?", 1, 20, 10)

# STEP 3 EXECUTION 
if st.button("Generate Recommendations"):
    unique_courses = df.drop_duplicates(subset="course_name").copy()

    # Generate recommendation score
    unique_courses["recommendation_score"] = (
        unique_courses["rating"] + np.random.normal(0, 0.1, len(unique_courses))
    )

    # Get Top N
    recommendations = unique_courses.nlargest(num_recommendations, "recommendation_score")

    # Select requested columns: course_id, recommendation_score, course_name, instructor, rating
    rec_display = recommendations[
        ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
    ].round(2)

    # Store in session state
    st.session_state.recommendations = rec_display
    st.session_state.course_options = rec_display["course_name"].tolist()

# STEP 3 DISPLAY
if "recommendations" in st.session_state:
    st.header("Step 3: Recommended Courses")
    st.dataframe(
        st.session_state.recommendations,
        use_container_width=True,
        hide_index=True
    )

# STEP 4 
if "recommendations" in st.session_state:
    st.header("Step 4: Select Courses")
    selected_courses = st.multiselect(
        "Choose courses to find the best instructors:",
        st.session_state.course_options
    )

    # STEP 5: High ranked course from selection   
if selected_courses:
    step5_result = df[
        (df["course_name"].isin(selected_courses)) &
        (df["rating"] >= 4)
    ].copy()

    if not step5_result.empty:
        # Pick highest rated instructor per course
        idx = step5_result.groupby("course_name")["rating"].idxmax()
        top_courses = step5_result.loc[idx]

        # Add recommendation score
        top_courses["recommendation_score"] = top_courses["rating"]

        # Sort for clean display
        top_courses = top_courses.sort_values(
            by="recommendation_score",
            ascending=False
        )

        # Select required columns
        step5_display = top_courses[
            ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
        ].round(2)

        st.header("Step 5: Highest Ranked Instructor for Each Selected Course")
        st.dataframe(
            step5_display,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No instructors found with a rating of 4 or higher for these selections.")
