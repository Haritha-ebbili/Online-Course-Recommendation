import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

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
st.caption("Model Output + Highest-Rated Similar Course Recommendations")

user_id = st.text_input(
    "Enter User ID",
    placeholder="Enter a valid User ID (e.g., 15796)"
)

num_recs = st.slider(
    "Number of course recommendations",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)

# ================= GENERATE TOP-N =================
if st.button("Generate Recommendations"):

    if not user_id.strip():
        st.warning("⚠️ Please enter a User ID")
        st.stop()

    uid = int(user_id) if user_id.isdigit() else user_id
    seen = user_history.get(uid, set())
    u_bias = user_bias.get(uid, 0)

    recs = courses[~courses["course_id"].isin(seen)].copy()

    # 🔹 MODEL BUILDING SCORE (EXACT)
    recs["recommendation_score"] = recs["course_id"].apply(
        lambda cid: global_mean + u_bias + item_bias.get(cid, 0)
    )

    top_recs = (
        recs.sort_values("recommendation_score", ascending=False)
            .head(num_recs)
            .reset_index(drop=True)
    )

    st.session_state["top_recs"] = top_recs
    st.session_state["shown_courses"] = set(top_recs["course_id"])

    st.subheader(f"Top Recommendations for User {uid}")
    st.dataframe(
        top_recs[
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

# ================= SIMILAR COURSES =================
if "top_recs" in st.session_state:

    selected_course = st.selectbox(
        "Select a course to get highest-rated similar courses",
        st.session_state["top_recs"]["course_name"]
    )

    if st.button("Recommend Highest-Rated Similar Courses"):

        selected_row = courses[courses["course_name"] == selected_course].iloc[0]
        selected_idx = course_id_to_index[selected_row["course_id"]]

        similarity_scores = list(enumerate(similarity_matrix[selected_idx]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        SIM_THRESHOLD = 0.30
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
                "course_id": cid,
                "course_name": courses.iloc[idx]["course_name"],
                "instructor": courses.iloc[idx]["instructor"],
                "rating": courses.iloc[idx]["rating"],
                "similarity_score": round(sim_score, 3)
            })

            if len(final_courses) == num_recs:
                break

        st.subheader("⭐ Highest-Rated Similar Courses")
        st.dataframe(
            pd.DataFrame(final_courses),
            use_container_width=True,
            height=300
        )
