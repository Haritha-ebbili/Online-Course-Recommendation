import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
@st.cache_resource
def build_courses(df):
    return (
        df.groupby("course_id", as_index=False)
          .agg({
              "course_name": "first",
              "instructor": "first",
              "rating": "mean"
          })
    )

courses = build_courses(df)

# ================= USER HISTORY =================
@st.cache_resource
def build_user_history(df):
    return df.groupby("user_id")["course_id"].apply(set).to_dict()

user_history = build_user_history(df)

# ================= CONTENT SIMILARITY =================
@st.cache_resource
def build_similarity(courses):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(courses["course_name"].fillna(""))
    return cosine_similarity(matrix)

similarity_matrix = build_similarity(courses)

course_id_to_index = {
    cid: idx for idx, cid in enumerate(courses["course_id"])
}

# ================= COLLABORATIVE FILTERING =================
global_mean = df["rating"].mean()
user_bias = (df.groupby("user_id")["rating"].mean() - global_mean).to_dict()
item_bias = (df.groupby("course_id")["rating"].mean() - global_mean).to_dict()

# ================= UI =================
st.title("🎓 Online Course Recommendation System")

user_id = st.text_input(
    "Enter User ID",
    placeholder="Enter User ID (e.g., 15796)"
)

num_recs = st.slider(
    "Number of course recommendations",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)

# ================= GENERATE PERSONALIZED RECOMMENDATIONS =================
if st.button("Generate Recommendations"):

    if not user_id.strip():
        st.warning("Please enter a User ID")
        st.stop()

    uid = int(user_id) if user_id.isdigit() else user_id
    seen_courses = user_history.get(uid, set())
    u_bias = user_bias.get(uid, 0)

    scores = []

    for _, row in courses.iterrows():
        cid = row["course_id"]

        if cid in seen_courses:
            continue

        # ---- similarity to user's own history ----
        if seen_courses:
            sim_vals = [
                similarity_matrix[course_id_to_index[cid]][course_id_to_index[sc]]
                for sc in seen_courses
                if sc in course_id_to_index
            ]
            sim_score = np.mean(sim_vals) if sim_vals else 0
        else:
            sim_score = 0

        final_score = (
            global_mean
            + u_bias
            + item_bias.get(cid, 0)
            + 0.7 * sim_score
        )

        scores.append({
            "course_id": cid,
            "recommendation_score": final_score,
            "course_name": row["course_name"],
            "instructor": row["instructor"],
            "rating": row["rating"]
        })

    rec_df = pd.DataFrame(scores)

    # 🔥 ENSURE UNIQUE COURSE NAMES
    rec_df = (
        rec_df.sort_values("recommendation_score", ascending=False)
              .drop_duplicates("course_name")
              .head(num_recs)
              .reset_index(drop=True)
    )

    st.subheader(f"Top Recommendations for User {uid}")
    st.dataframe(
        rec_df[
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
