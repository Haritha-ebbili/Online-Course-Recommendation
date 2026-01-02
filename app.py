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
}
.stButton > button {
    background: linear-gradient(45deg, #6a1b9a, #4527a0) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 30px !important;
    font-weight: 700 !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_pickle("full_data.pkl")

df = load_data()

# ================= TITLE =================
st.markdown('<h1 class="main-header">Course Recommendation System</h1>', unsafe_allow_html=True)

# ================= STEP 1 =================
st.header("Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

# ================= STEP 2 =================
st.header("Step 2: Number of Recommendations")
num_recommendations = st.slider("How many unique courses?", 1, 20, 10)

# ================= STEP 3 =================
if st.button("Generate Recommendations"):

    # Courses already taken by user
    user_taken_courses = df[df["user_id"] == user_id]["course_id"].unique()

    # Course-level aggregation (ENSURES UNIQUENESS)
    course_master = (
        df.groupby("course_id", as_index=False)
        .agg({
            "course_name": "first",
            "instructor": "first",
            "rating": "mean"
        })
    )

    # Remove courses already taken
    course_master = course_master[
        ~course_master["course_id"].isin(user_taken_courses)
    ]

    # Recommendation score
    course_master["recommendation_score"] = course_master["rating"]

    # Top-N UNIQUE courses
    recommendations = course_master.sort_values(
        by="recommendation_score",
        ascending=False
    ).head(num_recommendations)

    rec_display = recommendations[
        ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
    ].round(2)

    st.session_state.recommendations = rec_display
    st.session_state.course_options = rec_display["course_name"].tolist()

# ================= STEP 3 DISPLAY =================
if "recommendations" in st.session_state:
    st.header("Step 3: Unique Recommended Courses")
    st.dataframe(
        st.session_state.recommendations,
        use_container_width=True,
        hide_index=True
    )

# ================= STEP 4 =================
if "recommendations" in st.session_state:
    st.header("Step 4: Select Courses")
    selected_courses = st.multiselect(
        "Choose courses to find the best instructor:",
        st.session_state.course_options
    )

    # ================= STEP 5 =================
    if selected_courses:
        step5_df = df[
            (df["course_name"].isin(selected_courses)) &
            (df["rating"] >= 4)
        ].copy()
