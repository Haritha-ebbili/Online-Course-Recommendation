import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

# ================= ALLOWED UNIQUE COURSES =================
ALLOWED_COURSES = {
    "Python for Beginners",
    "Cybersecurity for Professionals",
    "DevOps and Continuous Deployment",
    "Project Management Fundamentals",
    "Ethical Hacking Masterclass",
    "Networking and System Administration",
    "Personal Finance and Wealth Building",
    "Blockchain and Decentralized Applications",
    "Graphic Design with Canva",
    "Fitness and Nutrition Coaching",
    "Public Speaking Mastery",
    "Photography and Video Editing",
    "Advanced Machine Learning",
    "Game Development with Unity",
    "Cloud Computing Essentials",
    "Mobile App Development with Swift",
    "Data Visualization with Tableau",
    "Stock Market and Trading Strategies",
    "Fundamentals of Digital Marketing",
    "AI for Business Leaders"
}

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_excel("online_course_recommendation.xlsx")
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_data()

# ================= VALIDATION =================
required_cols = {"user_id", "course_id", "course_name", "instructor", "rating"}
if not required_cols.issubset(df.columns):
    st.error(f"Dataset must contain columns: {required_cols}")
    st.stop()

# ================= UNIQUE COURSE MASTER =================
courses = (
    df.groupby("course_id", as_index=False)
      .agg({
          "course_name": "first",
          "instructor": "first",
          "rating": "mean"
      })
)

# Keep ONLY your 20 unique courses
courses = courses[courses["course_name"].isin(ALLOWED_COURSES)].reset_index(drop=True)

# ================= USER HISTORY =================
user_history = (
    df[df["course_name"].isin(ALLOWED_COURSES)]
    .groupby("user_id")["course_id"]
    .apply(set)
    .to_dict()
)

# ================= CONTENT-BASED SIMILARITY =================
@st.cache_data
def build_similarity(course_names):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(course_names)
    return cosine_similarity(matrix)

similarity_matrix = build_similarity(courses["course_name"])

course_id_to_index = {
    cid: idx for idx, cid in enumerate(courses["course_id"])
}

# ================= UI =================
st.title("🎓 Online Course Recommendation System")
st.caption("Unique • High-Quality • Deployment Ready")

user_id = st.text_input(
    "Enter User ID",
    value=str(df["user_id"].iloc[0])
)

num_recommendations = st.slider(
    "Number of course recommendations",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

# ================= INITIAL RECOMMENDATIONS =================
if st.button("Generate Recommendations"):

    uid = int(user_id) if user_id.isdigit() else user_id
    seen_courses = user_history.get(uid, set())

    initial_df = (
        courses[~courses["course_id"].isin(seen_courses)]
        .sort_values("rating", ascending=False)
        .head(num_recommendations)
        .reset_index(drop=True)
    )

    st.session_state["shown_courses"] = set(initial_df["course_id"])
    st.session_state["initial_df"] = initial_df

# ================= DISPLAY INITIAL =================
if "initial_df" in st.session_state:

    st.subheader("📌 Recommended Courses (Unique)")
    st.dataframe(
        st.session_state["initial_df"],
        use_container_width=True,
        height=300
    )

    selected_course = st.selectbox(
        "Select a course to get similar recommendations",
        st.session_state["initial_df"]["course_name"]
    )

    # ================= SIMILAR RECOMMENDATIONS =================
    if st.button("Recommend Similar Courses"):

        selected_row = courses[courses["course_name"] == selected_course].iloc[0]
        selected_idx = course_id_to_index[selected_row["course_id"]]

        similarity_scores = list(enumerate(similarity_matrix[selected_idx]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        SIM_THRESHOLD = 0.30
        final_courses = []

        for idx, score in similarity_scores:

            cid = courses.iloc[idx]["course_id"]

            if idx == selected_idx:
                continue

            if cid in st.session_state["shown_courses"]:
                continue

            if score < SIM_THRESHOLD:
                break

            final_courses.append({
                "course_id": cid,
                "course_name": courses.iloc[idx]["course_name"],
                "instructor": courses.iloc[idx]["instructor"],
                "rating": courses.iloc[idx]["rating"],
                "similarity_score": round(score, 3)
            })

            st.session_state["shown_courses"].add(cid)

            if len(final_courses) == num_recommendations:
                break

        similar_df = pd.DataFrame(final_courses)

        st.subheader("⭐ Highest-Rated Similar Courses")
        st.dataframe(
            similar_df,
            use_container_width=True,
            height=300
        )
