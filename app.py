import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Online Course Recommendation System", layout="wide")

# ================= LOAD DATA =================
@st.cache_resource
def load_data():
    df = pd.read_excel("online_course_recommendation.xlsx")
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_data()

# ================= COURSE MASTER =================
courses = (
    df.groupby("course_id", as_index=False)
      .agg({
          "course_name": "first",
          "instructor": "first",
          "rating": "mean"
      })
)

# ================= USER–COURSE RATINGS =================
user_course_ratings = (
    df.groupby(["user_id", "course_id"])["rating"]
      .mean()
      .reset_index()
)

# ================= TF-IDF COURSE VECTORS =================
@st.cache_resource
def build_tfidf(courses):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(courses["course_name"].fillna(""))
    return tfidf, vectors

tfidf, course_vectors = build_tfidf(courses)

course_id_to_index = {
    cid: idx for idx, cid in enumerate(courses["course_id"])
}

# ================= UI =================
st.title("🎓 Online Course Recommendation System")

user_id = st.text_input(
    "Enter User ID",
    placeholder="Enter User ID (e.g., 15796)"
)

num_recs = st.slider(
    "Number of course recommendations",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# ================= GENERATE PERSONALIZED RECOMMENDATIONS =================
if st.button("Generate Recommendations"):

    if not user_id.strip():
        st.warning("Please enter a User ID")
        st.stop()

    uid = int(user_id) if user_id.isdigit() else user_id
    user_data = user_course_ratings[user_course_ratings["user_id"] == uid]

    # ---------- COLD START ----------
    if user_data.empty:
        st.info("Cold-start user detected. Showing popular courses.")
        popular = (
            courses.sort_values("rating", ascending=False)
                   .drop_duplicates("course_name")
                   .head(num_recs)
                   .reset_index(drop=True)
        )
        st.dataframe(popular, use_container_width=True)
        st.stop()

    # ---------- BUILD USER PROFILE ----------
    user_mean = user_data["rating"].mean()
    liked_courses = user_data[user_data["rating"] >= user_mean]["course_id"]

    liked_indices = [
        course_id_to_index[cid]
        for cid in liked_courses
        if cid in course_id_to_index
    ]

    if not liked_indices:
        st.info("Not enough preference data. Showing popular courses.")
        popular = (
            courses.sort_values("rating", ascending=False)
                   .drop_duplicates("course_name")
                   .head(num_recs)
                   .reset_index(drop=True)
        )
        st.dataframe(popular, use_container_width=True)
        st.stop()

    # 🔥 FINAL FIX (dense 2D array)
    user_profile = course_vectors[liked_indices].mean(axis=0)
    user_profile = np.asarray(user_profile).astype(np.float64)
    user_profile = user_profile.reshape(1, -1)

    # 🔥 IMPORTANT: user_profile FIRST
    similarity_scores = cosine_similarity(user_profile, course_vectors)[0]

    courses["recommendation_score"] = similarity_scores

    # ---------- REMOVE SEEN COURSES ----------
    courses_filtered = courses[
        ~courses["course_id"].isin(user_data["course_id"])
    ]

    # ---------- FINAL TOP-N UNIQUE ----------
    final_recs = (
        courses_filtered
        .sort_values("recommendation_score", ascending=False)
        .drop_duplicates("course_name")
        .head(num_recs)
        .reset_index(drop=True)
    )

    st.subheader(f"Top Recommendations for User {uid}")
    st.dataframe(
        final_recs[
            [
                "course_id",
                "recommendation_score",
                "course_name",
                "instructor",
                "rating"
            ]
        ],
        use_container_width=True,
        height=300
    )
