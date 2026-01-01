import streamlit as st
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Course Recommender", layout="wide")

# ================= CUSTOM CSS =================
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

.stSlider > div > div > div > div {
    background-color: #7b1fa2 !important;
    height: 10px !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: linear-gradient(45deg, #1e3c72, #7b1fa2) !important;
    color: white !important;
    border-radius: 50px !important;
    padding: 15px 40px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_pickle("full_data.pkl")

df = load_data()

# ================= TITLE =================
st.markdown(
    '<h1 class="main-header">Course Recommendation System</h1>',
    unsafe_allow_html=True
)

# ================= STEP 1 =================
st.header("Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, value=15796)

# ================= STEP 2 =================
st.header("Step 2: Number of Recommendations")
num_recommendations = st.slider("How many unique courses?", 1, 20, 10)

# ================= STEP 3 EXECUTION (USER-BASED, NO RANDOMNESS) =================
if st.button("Generate Recommendations"):

    # Courses already taken by this user
    user_courses = df[df["user_id"] == user_id]["course_name"].unique()

    # Candidate courses (exclude taken ones)
    candidate_courses = df[~df["course_name"].isin(user_courses)]

    # Unique course names
    unique_courses = candidate_courses.drop_duplicates(subset="course_name")

    # Deterministic recommendation score (dataset-based)
    unique_courses["recommendation_score"] = unique_courses["rating"]

    # Top-N recommendations
    recommendations = unique_courses.sort_values(
        by="recommendation_score", ascending=False
    ).head(num_recommendations)

    rec_display = recommendations[
        ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
    ]

    # Store results
    st.session_state.recommendations = rec_display
    st.session_state.course_options = rec_display["course_name"].tolist()

# ================= STEP 3 DISPLAY (ALWAYS VISIBLE) =================
if "recommendations" in st.session_state:
    st.header("Step 3: Recommended Courses (User-Specific)")
    st.dataframe(
        st.session_state.recommendations,
        use_container_width=True,
        hide_index=True
    )

# ================= STEP 4 =================
if "recommendations" in st.session_state:
    st.header("Step 4: Select Courses")
    selected_courses = st.multiselect(
        "Choose courses:",
        st.session_state.course_options
    )

    # ================= STEP 5 =================
    if selected_courses:
        step5_result = df[
            (df["course_name"].isin(selected_courses)) &
            (df["rating"] >= 4) &
            (df["rating"] <= 5)
        ][["course_id", "course_name", "instructor", "rating"]].drop_duplicates()

        # Step-5 score = rating (refinement only)
        step5_result["recommendation_score"] = step5_result["rating"]

        step5_result = step5_result[
            ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
        ].sort_values(
            by=["course_name", "recommendation_score"],
            ascending=[True, False]
        )

        st.header("Step 5: Selected Courses (Rating 4–5)")
        st.dataframe(
            step5_result,
            use_container_width=True,
            hide_index=True
        )
