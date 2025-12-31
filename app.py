import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data():
    df = pd.read_excel("online_course_recommendation.xlsx")
    return df

df = load_data()

# ---------------- CLEAN COLUMNS ----------------
df.columns = df.columns.str.lower().str.strip()

required_cols = {"user_id", "course_id", "course_name", "instructor", "rating"}
if not required_cols.issubset(df.columns):
    st.error(f"Dataset must contain columns: {required_cols}")
    st.stop()

# ---------------- COURSE MASTER TABLE ----------------
courses = (
    df[["course_id", "course_name", "instructor", "rating"]]
    .drop_duplicates("course_id")
    .reset_index(drop=True)
)

# ---------------- COLLABORATIVE FILTERING ----------------
global_mean = df["rating"].mean()

user_bias = (df.groupby("user_id")["rating"].mean() - global_mean).to_dict()
item_bias = (df.groupby("course_id")["rating"].mean() - global_mean).to_dict()
user_history = df.groupby("user_id")["course_id"].apply(list).to_dict()

# ---------------- CONTENT-BASED SIMILARITY ----------------
@st.cache_resource
def compute_similarity(course_names):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(course_names.fillna(""))
    return cosine_similarity(tfidf_matrix)

similarity_matrix = compute_similarity(courses["course_name"])

course_id_to_index = {
    cid: idx for idx, cid in enumerate(courses["course_id"])
}

# ---------------- UI ----------------
st.title("🎓 Online Course Recommendation System")
st.caption("Highest-Rated Similar Course Recommendation")

user_input = st.text_input(
    "Enter User ID",
    value=str(df["user_id"].iloc[0])
)

num_recommendations = st.slider(
    "Number of course recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=5
)

# ---------------- INITIAL RECOMMENDATIONS ----------------
if st.button("Generate Recommendations"):

    user_id = int(user_input) if user_input.isdigit() else user_input
    history = user_history.get(user_id, [])
    u_bias = user_bias.get(user_id, 0)

    available = courses[~courses["course_id"].isin(history)].copy()

    user_mean = (
        df[df["course_id"].isin(history)]["rating"].mean()
        if history else df["rating"].mean()
    )

    available["score"] = (
        global_mean
        + u_bias
        + available["course_id"].map(item_bias).fillna(0)
        + 0.3 * (available["rating"] - user_mean)
    )

    initial_df = (
        available
        .sort_values("score", ascending=False)
        .head(num_recommendations)
        .reset_index(drop=True)
    )

    st.session_state["initial_df"] = initial_df
    st.session_state["shown_courses"] = set(initial_df["course_id"])

# ---------------- DISPLAY INITIAL ----------------
if "initial_df" in st.session_state:

    st.subheader("📌 Recommended Courses")
    st.dataframe(
        st.session_state["initial_df"][
            ["course_id", "course_name", "instructor", "rating"]
        ],
        height=400,
        use_container_width=True
    )

    selected_course = st.selectbox(
        "Select a course to get highest-rated similar courses",
        st.session_state["initial_df"]["course_name"]
    )

    # ---------------- BEST SIMILAR COURSES ----------------
    if st.button("Recommend Highest-Rated Similar Courses"):

        selected_row = courses[courses["course_name"] == selected_course].iloc[0]
        selected_idx = course_id_to_index[selected_row["course_id"]]

        similarity_scores = list(enumerate(similarity_matrix[selected_idx]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        TOP_K = 5
        SIM_THRESHOLD = 0.35

        filtered = []

        for idx, sim_score in similarity_scores:
            cid = courses.iloc[idx]["course_id"]

            if idx == selected_idx:
                continue

            if cid in st.session_state["shown_courses"]:
                continue

            if sim_score < SIM_THRESHOLD:
                break

            filtered.append({
                "index": idx,
                "similarity": sim_score,
                "rating": courses.iloc[idx]["rating"]
            })

        filtered = sorted(
            filtered,
            key=lambda x: (x["similarity"], x["rating"]),
            reverse=True
        )[:TOP_K]

        best_indices = [item["index"] for item in filtered]

        similar_df = courses.iloc[best_indices].copy()
        similar_df["similarity_score"] = [
            similarity_matrix[selected_idx][i] for i in best_indices
        ]

        similar_df = similar_df.sort_values(
            by=["similarity_score", "rating"],
            ascending=[False, False]
        )

        st.subheader("⭐ Highest-Rated Similar Courses")
        st.dataframe(
            similar_df[
                ["course_id", "course_name", "instructor", "rating", "similarity_score"]
            ],
            height=350,
            use_container_width=True
        )
