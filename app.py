import streamlit as st
import pandas as pd
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
    return pd.read_excel("online_course_recommendation.xlsx")

df = load_data()

# ---------------- CLEAN COLUMNS ----------------
df.columns = df.columns.str.lower().str.strip()

required_cols = {"user_id", "course_id", "course_name", "instructor", "rating"}
if not required_cols.issubset(df.columns):
    st.error(f"Dataset must contain columns: {required_cols}")
    st.stop()

# ---------------- UNIQUE COURSE MASTER ----------------
courses = (
    df.groupby("course_id", as_index=False)
      .agg({
          "course_name": "first",
          "instructor": "first",
          "rating": "mean"
      })
)

# ---------------- COLLABORATIVE DATA ----------------
global_mean = df["rating"].mean()
user_bias = (df.groupby("user_id")["rating"].mean() - global_mean).to_dict()
item_bias = (df.groupby("course_id")["rating"].mean() - global_mean).to_dict()
user_history = df.groupby("user_id")["course_id"].apply(set).to_dict()

# ---------------- CONTENT SIMILARITY ----------------
@st.cache_resource
def compute_similarity(names):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(names.fillna(""))
    return cosine_similarity(matrix)

similarity_matrix = compute_similarity(courses["course_name"])

course_id_to_index = {
    cid: idx for idx, cid in enumerate(courses["course_id"])
}

# ---------------- UI ----------------
st.title("🎓 Online Course Recommendation System")
st.caption("Unique + Highest-Rated Similar Course Recommendations")

user_input = st.text_input(
    "Enter User ID",
    value=str(df["user_id"].iloc[0])
)

num_recommendations = st.slider(
    "Number of course recommendations",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# ---------------- INITIAL RECOMMENDATIONS ----------------
if st.button("Generate Recommendations"):

    user_id = int(user_input) if user_input.isdigit() else user_input
    history = user_history.get(user_id, set())
    u_bias = user_bias.get(user_id, 0)

    available = courses[~courses["course_id"].isin(history)].copy()

    available["score"] = (
        global_mean
        + u_bias
        + available["course_id"].map(item_bias).fillna(0)
        + 0.3 * (available["rating"] - global_mean)
    )

    initial_df = (
        available
        .sort_values("score", ascending=False)
        .drop_duplicates("course_id")
        .head(num_recommendations)
        .reset_index(drop=True)
    )

    st.session_state["shown_courses"] = set(initial_df["course_id"])
    st.session_state["initial_df"] = initial_df

# ---------------- DISPLAY INITIAL ----------------
if "initial_df" in st.session_state:

    st.subheader("📌 Recommended Courses (Unique)")
    st.dataframe(
        st.session_state["initial_df"][
            ["course_id", "course_name", "instructor", "rating"]
        ],
        use_container_width=True,
        height=350
    )

    selected_course = st.selectbox(
        "Select a course to get highest-rated similar courses",
        st.session_state["initial_df"]["course_name"]
    )

    # ---------------- UNIQUE SIMILAR COURSES ----------------
    if st.button("Recommend Highest-Rated Similar Courses"):

        selected_row = courses[courses["course_name"] == selected_course].iloc[0]
        selected_idx = course_id_to_index[selected_row["course_id"]]

        similarity_scores = list(enumerate(similarity_matrix[selected_idx]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        TOP_K = num_recommendations
        SIM_THRESHOLD = 0.35

        final_courses = []

        for idx, sim_score in similarity_scores:
            cid = courses.iloc[idx]["course_id"]

            if idx == selected_idx:
                continue

            if cid in st.session_state["shown_courses"]:
                continue

            if sim_score < SIM_THRESHOLD:
                break

            final_courses.append({
                "index": idx,
                "similarity": sim_score,
                "rating": courses.iloc[idx]["rating"]
            })

            st.session_state["shown_courses"].add(cid)

            if len(final_courses) == TOP_K:
                break

        similar_df = courses.iloc[[c["index"] for c in final_courses]].copy()
        similar_df["similarity_score"] = [c["similarity"] for c in final_courses]

        similar_df = similar_df.sort_values(
            by=["similarity_score", "rating"],
            ascending=[False, False]
        )

        st.subheader("⭐ Highest-Rated Similar Courses (100% Unique)")
        st.dataframe(
            similar_df[
                ["course_id", "course_name", "instructor", "rating", "similarity_score"]
            ],
            use_container_width=True,
            height=350
        )
