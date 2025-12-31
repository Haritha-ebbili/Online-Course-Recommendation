import streamlit as st
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Course Recommendation System",
    layout="wide"
)

st.title("🎓 Online Course Recommendation System")
st.write("Popularity-Based Model (Lowest RMSE)")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_excel("/mnt/data/online_course_recommendation.xlsx")
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_data()

# ================= USER INPUT =================
user_id = st.text_input("Enter User ID:", value="15796")

if user_id.isdigit():
    user_id = int(user_id)

TOP_N = 5

# ================= POPULARITY MODEL =================
# Popularity score = mean rating of each course
course_popularity = (
    df.groupby("course_id", as_index=False)
      .agg(
          recommendation_score=("rating", "mean"),
          course_name=("course_name", "first"),
          instructor=("instructor", "first"),
          rating=("rating", "mean")
      )
)

# ================= USER-SPECIFIC FILTER =================
# Courses already taken by the user
user_courses = df[df["user_id"] == user_id]["course_id"].unique()

# Recommend only unseen courses
recommendations = (
    course_popularity[
        ~course_popularity["course_id"].isin(user_courses)
    ]
    .sort_values("recommendation_score", ascending=False)
    .head(TOP_N)
    .reset_index(drop=True)
)

# ================= DISPLAY OUTPUT =================
st.subheader(f"Top {TOP_N} Recommendations for User {user_id}")

if recommendations.empty:
    st.warning("No recommendations available for this user.")
else:
    st.dataframe(
        recommendations[
            ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
        ],
        use_container_width=True
    )
