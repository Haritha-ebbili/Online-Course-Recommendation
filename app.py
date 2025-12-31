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

# ================= TF-IDF =================
@st.cache_resource
def build_tfidf(course_names):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(course_names.fillna(""))
    return vectors

course_vectors = build_tfidf(courses["course_name"])

course_id_to_index = {
    cid: idx for idx, cid in enumerate(courses["course_id"])
}

index_to_course_id = {
    idx: cid for cid, idx in course_id_to_index.items()
}

# ================= UI =================
st.title("🎓 Online Course Recommendation System")

user_id = st.text_input("Enter User ID", placeholder="e.g., 15796")

num_recs = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# ================= GENERATE RECOMMENDATIONS =================
if st.button("Generate Recommendations"):

    uid = int(user_id)
    user_data = user_course_ratings[user_course_ratings["user_id"] == uid]

    # ---------- COLD START ----------
    if user_data.empty:
        st.warning("Cold-start user. Showing popular courses.")
        popular = (
            courses.sort_values(["rating", "course_id"], ascending=[False, True])
                   .head(num_recs)
        )
        st.dataframe(popular)
        st.stop()

    # ---------- USER PROFILE ----------
    user_mean = user_data["rating"].mean()
    liked_courses = user_data[user_data["rating"] >= user_mean]["course_id"]

    liked_indices = [
        course_id_to_index[cid]
        for cid in liked_courses
        if cid in course_id_to_index
    ]

    if not liked_indices:
        st.warning("Not enough preferences. Showing popular courses.")
        st.dataframe(
            courses.sort_values("rating", ascending=False).head(num_recs)
        )
        st.stop()

    liked_vectors = course_vectors[liked_indices].toarray()
    user_profile = np.mean(liked_vectors, axis=0).reshape(1, -1)

    # ---------- COSINE SIMILARITY ----------
    similarity_scores = cosine_similarity(user_profile, course_vectors)[0]
    courses["recommendation_score"] = similarity_scores

    # ---------- FILTER SEEN COURSES ----------
    courses_filtered = courses[
        ~courses["course_id"].isin(user_data["course_id"])
    ]

    final_recs = (
        courses_filtered
        .sort_values(
            ["recommendation_score", "rating"],
            ascending=[False, False]
        )
        .head(num_recs)
        .reset_index(drop=True)
    )

    st.subheader(f"Top Recommendations for User {uid}")
    st.dataframe(final_recs)

    # ================= SELECT A COURSE =================
    st.subheader("🔍 Find Similar High-Rated Courses")

    selected_course = st.selectbox(
        "Select a recommended course",
        final_recs["course_name"].tolist()
    )

    if selected_course:
        selected_course_id = final_recs[
            final_recs["course_name"] == selected_course
        ]["course_id"].values[0]

        selected_idx = course_id_to_index[selected_course_id]
        selected_vector = course_vectors[selected_idx]

        similarity_with_selected = cosine_similarity(
            selected_vector, course_vectors
        )[0]

        courses["similarity"] = similarity_with_selected

        selected_rating = courses.loc[
            courses["course_id"] == selected_course_id, "rating"
        ].values[0]

        similar_high_rated = (
            courses[
                (courses["rating"] > selected_rating) &
                (courses["course_id"] != selected_course_id)
            ]
            .sort_values(
                ["similarity", "rating"],
                ascending=[False, False]
            )
            .head(5)
            .reset_index(drop=True)
        )

        st.subheader(f"📈 Courses Similar to '{selected_course}' with Higher Ratings")
        st.dataframe(similar_high_rated)
