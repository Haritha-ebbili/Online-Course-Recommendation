import streamlit as st
import pandas as pd
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Course Recommender", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* FULL PAGE BACKGROUND COLOR */
.stApp {
    background-color: #D5CABD !important;
}

/* MAIN HEADER */
.main-header {
    font-size: 3.5rem !important;
    background: linear-gradient(90deg, #6a1b9a, #ec407a) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
}

/* SLIDER */
.stSlider > div > div > div > div {
    background-color: #7b1fa2 !important;
    height: 10px !important;
    border-radius: 12px !important;
}

/* BUTTON */
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

/* MULTISELECT */
.stMultiSelect > div > div > div {
    border: 3px solid #7b1fa2 !important;
    border-radius: 15px !important;
    background: #f3e5f5 !important;
}

/* TABLE */
.stDataFrame table {
    border-radius: 15px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
}

.stDataFrame thead tr th {
    background: #1976d2 !important;
    color: white !important;
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
st.markdown('<h1 class="main-header">Course Recommendation System</h1>', unsafe_allow_html=True)

# ================= STEP 1 =================
st.header("Step 1: Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=49999, value=15796)

# ================= STEP 2 =================
st.header("Step 2: Number of Recommendations")
num_recommendations = st.slider("How many unique courses?", 1, 20, 10)

# ================= STEP 3 EXECUTION =================
if st.button("Generate Recommendations"):
    unique_courses = df.drop_duplicates(subset="course_name")

    unique_courses["score"] = (
        unique_courses["rating"] + np.random.normal(0, 0.1, len(unique_courses))
    )

    recommendations = unique_courses.nlargest(num_recommendations, "score")

    rec_display = recommendations[
        ["course_name", "instructor", "rating", "course_price", "score"]
    ].round(2)

    rec_display.columns = [
        "Course Name",
        "Instructor",
        "Rating",
        "Course Price",
        "Pred Score"
    ]

    # Store in session state
    st.session_state.recommendations = rec_display
    st.session_state.course_options = rec_display["Course Name"].tolist()

# ================= STEP 3 DISPLAY =================
if "recommendations" in st.session_state:
    st.header("Step 3: Recommended Courses")

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
        ][["course_name", "instructor", "rating", "course_price"]].drop_duplicates()

        step5_result["pred_score"] = step5_result["rating"]

        step5_result = step5_result.sort_values(
            by=["course_name", "pred_score"],
            ascending=[True, False]
        )

        step5_result.columns = [
            "Course Name",
            "Instructor",
            "Rating",
            "Course Price",
            "Pred Score"
        ]

        st.header("Step 5: Selected Course with Different Instructors (Rating 4–5)")
        st.dataframe(
            step5_result,
            use_container_width=True,
            hide_index=True
        )
