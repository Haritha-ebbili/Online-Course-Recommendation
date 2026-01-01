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
    if os.path.exists("full_data.pkl"):
        return pd.read_pickle("full_data.pkl")
    else:
        # fallback demo data
        data = {
            'user_id': np.random.randint(15790, 15800, 100),
            'course_id': np.random.randint(100, 500, 100),
            'course_name': [f"Course {i}" for i in range(100)],
            'instructor': [f"Instructor {i}" for i in range(100)],
            'rating': np.random.uniform(3.5, 5.0, 100)
        }
        return pd.DataFrame(data)

df = load_data()

# ================= SESSION STATE =================
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "clicked" not in st.session_state:
    st.session_state.clicked = False

# ================= TITLE =================
st.markdown('<h1 class="main-header">Course Recommendation System</h1>', unsafe_allow_html=True)

# ================= STEP 1 & STEP 2 =================
col1, col2 = st.columns(2)

with col1:
    st.header("Step 1: User Profile")
    user_id_input = st.number_input("Enter User ID", min_value=1, value=15796)

with col2:
    st.header("Step 2: Preferences")
    num_recs = st.slider("Number of recommendations", 1, 20, 10)

# ================= GENERATE RECOMMENDATIONS =================
if st.button("Generate Recommendations"):

    # 1️⃣ Courses already taken by the user
    user_history = df[df["user_id"] == user_id_input]["course_name"].unique()

    # 2️⃣ Remove already taken courses
    candidate_courses = df[~df["course_name"].isin(user_history)]

    # 3️⃣ Keep ONLY high-quality courses (rating 4–5)
    candidate_courses = candidate_courses[
        (candidate_courses["rating"] >= 4) &
        (candidate_courses["rating"] <= 5)
    ]

    # 4️⃣ Unique course names with aggregated rating
    recommendations = (
        candidate_courses
        .groupby("course_name", as_index=False)
        .agg({
            "course_id": "first",
            "instructor": "first",
            "rating": "mean"
        })
    )

    # 5️⃣ Rank courses (best first)
    recommendations = recommendations.sort_values(
        by="rating", ascending=False
    ).head(num_recs)

    # 6️⃣ Recommendation score = ranking score (4–5)
    recommendations["recommendation_score"] = recommendations["rating"].round(2)

    st.session_state.recommendations = recommendations
    st.session_state.clicked = True

# ================= STEP 3 =================
if st.session_state.clicked and st.session_state.recommendations is not None:
    st.divider()
    st.header("Step 3: Recommended Courses (Rating 4–5 Only)")

    display_df = st.session_state.recommendations[
        ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
    ]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ================= STEP 4 =================
    st.header("Step 4: Select Courses to Compare")
    course_list = display_df["course_name"].tolist()
    selected_courses = st.multiselect("Choose courses:", course_list)

    # ================= STEP 5 =================
    if selected_courses:
        st.header("Step 5: Selected Courses (High-Quality Ranking)")

        step5_result = df[
            (df["course_name"].isin(selected_courses)) &
            (df["rating"] >= 4) &
            (df["rating"] <= 5)
        ][["course_id", "course_name", "instructor", "rating"]]

        step5_result = step5_result.drop_duplicates(subset="course_name")
        step5_result = step5_result.sort_values(
            by="rating", ascending=False
        )

        step5_result["recommendation_score"] = step5_result["rating"].round(2)

        st.dataframe(
            step5_result[
                ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Select courses in Step 4 to view ranked details.")
