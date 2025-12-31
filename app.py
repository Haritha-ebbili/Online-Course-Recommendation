import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hybrid Course Recommender", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("data.csv")
        return df
    except Exception as e:
        st.error(f"Failed to load data.csv: {e}")
        st.stop()

df = load_data()

# ---------------- PREPARE DATA ----------------
courses = (
    df[["course_id", "course_name", "instructor", "rating"]]
    .drop_duplicates("course_id")
    .reset_index(drop=True)
)

# ---------------- COLLABORATIVE FILTERING ----------------
global_mean = df["rating"].mean()

user_bias = (
    df.groupby("user_id")["rating"].mean() - global_mean
).to_dict()

item_bias = (
    df.groupby("course_id")["rating"].mean() - global_mean
).to_dict()

user_history = (
    df.groupby("user_id")["course_id"].apply(list).to_dict()
)

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
st.title("🎓 Hybrid Course Recommendation System")
st.caption("No Pickle • Scrollable • Interactive • Duplicate-Free")

user_input = st.text_input("Enter User ID", value="15796")

num_recommendations = st.slider(
    "How many courses do you want to recommend?",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

# ---------------- INITIAL RECOMMENDATION ----------------
if st.button("Generate Recommendations"):

    user_id = int(user_input) if user_input.isdigit() else user_input

    u_bias = user_bias.get(user_id, 0)
    history = user_history.get(user_id, [])

    available = courses[~courses["course_id"].isin(history)].copy()

    user_mean = (
        df[df["course_id"].isin(history)]["rating"].mean()
        if history else df["rating"].mean()
    )

    available["score"] = available["course_id"].apply(
        lambda cid: (
            global_mean
            + u_bias
            + item_bias.get(cid, 0)
        )
    ) + 0.3 * (available["rating"] - user_mean)

    initial_df = (
        available
        .sort_values("score", ascending=False)
        .head(num_recommendations)
        .reset_index(drop=True)
    )

    st.session_state["shown_courses"] = set(initial_df["course_id"])
    st.session_state["initial_df"] = initial_df

# ---------------- DISPLAY INITIAL RESULTS ----------------
if "initial_df" in st.session_state:

    st.subheader("📌 Recommended Courses (Scrollable)")
    st.dataframe(
        st.session_state["initial_df"][
            ["course_id", "course_name", "instructor", "rating"]
        ],
        height=400,
        use_container_width=True
    )

    selected_course = st.selectbox(
        "Select a course to get similar recommendations",
        st.session_state["initial_df"]["course_name"]
    )

    # ---------------- SIMILAR RECOMMENDATION ----------------
    if st.button("Recommend Similar Courses"):

        selected_row = courses[courses["course_name"] == selected_course].iloc[0]
        selected_id = selected_row["course_id"]
        selected_idx = course_id_to_index[selected_id]

        similarity_scores = list(enumerate(similarity_matrix[selected_idx]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        indices = []
        for idx, _ in similarity_scores:
            cid = courses.iloc[idx]["course_id"]
            if cid not in st.session_state["shown_courses"]:
                indices.append(idx)
                st.session_state["shown_courses"].add(cid)
            if len(indices) == num_recommendations:
                break

        similar_df = courses.iloc[indices].copy()
        similar_df["similarity_score"] = [
            similarity_matrix[selected_idx][i] for i in indices
        ]

        st.subheader("🔁 Similar Course Recommendations")
        st.dataframe(
            similar_df[
                ["course_id", "course_name", "instructor", "rating", "similarity_score"]
            ],
            height=400,
            use_container_width=True
        )
