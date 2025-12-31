import streamlit as st
import pandas as pd
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

user_bias = (
    df.groupby("user_id")["rating"].mean() - global_mean
).to_dict()

item_bias = (
    df.groupby("course_id")["rating"].mean() - global_mean
).to_dict()

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

# ================= GENERATE RECOMMENDATIONS =================
if st.button("Generate Recommendations"):

    if not user_id.strip():
        st.warning("Please enter a User ID")
        st.stop()

    uid = int(user_id) if user_id.isdigit() else user_id
    seen_courses = user_history.get(uid, set())
    u_bias = user_bias.get(uid, 0)

    # ---------- STEP 1: BASE SCORE ----------
    df_scores = df.copy()
    df_scores["base_score"] = (
        global_mean
        + u_bias
        + df_scores["course_id"].map(item_bias).fillna(0)
    )

    # ---------- STEP 2: USER HISTORY BOOST ----------
    if seen_courses:
        seen_indices = [
            course_id_to_index[cid]
            for cid in seen_courses
            if cid in course_id_to_index
        ]

        if seen_indices:
            similarity_boost = similarity_matrix[seen_indices].mean(axis=0)
            df_scores["similarity_boost"] = df_scores["course_id"].map(
                lambda cid: similarity_boost[course_id_to_index[cid]]
                if cid in course_id_to_index else 0
            )
        else:
            df_scores["similarity_boost"] = 0
    else:
        df_scores["similarity_boost"] = 0

    # ---------- STEP 3: FINAL SCORE ----------
    df_scores["recommendation_score"] = (
        df_scores["base_score"] + 0.5 * df_scores["similarity_boost"]
    )

    # ---------- STEP 4: REMOVE SEEN COURSES ----------
    df_scores = df_scores[~df_scores["course_id"].isin(seen_courses)]

    # ---------- STEP 5: UNIQUE COURSE NAMES ----------
    final_recs = (
        df_scores
        .groupby("course_name", as_index=False)
        .agg({
            "course_id": "first",
            "recommendation_score": "max",
            "instructor": "first",
            "rating": "mean"
        })
        .sort_values("recommendation_score", ascending=False)
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
