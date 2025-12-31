import streamlit as st
import pandas as pd

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
courses = (
    df.groupby("course_id", as_index=False)
      .agg({
          "course_name": "first",
          "instructor": "first",
          "rating": "mean"
      })
)

# ================= COLLABORATIVE FILTERING =================
global_mean = df["rating"].mean()

user_bias = (
    df.groupby("user_id")["rating"].mean() - global_mean
).to_dict()

item_bias = (
    df.groupby("course_id")["rating"].mean() - global_mean
).to_dict()

user_history = (
    df.groupby("user_id")["course_id"]
      .apply(set)
      .to_dict()
)

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

# ================= GENERATE UNIQUE TOP-N =================
if st.button("Generate Recommendations"):

    if not user_id.strip():
        st.warning("Please enter a User ID")
        st.stop()

    uid = int(user_id) if user_id.isdigit() else user_id
    seen = user_history.get(uid, set())
    u_bias = user_bias.get(uid, 0)

    recs = courses[~courses["course_id"].isin(seen)].copy()

    # ✅ MODEL BUILDING SCORE
    recs["recommendation_score"] = recs["course_id"].apply(
        lambda cid: global_mean + u_bias + item_bias.get(cid, 0)
    )

    # ✅ CRITICAL FIX: UNIQUE COURSE NAMES
    top_recs = (
        recs.sort_values("recommendation_score", ascending=False)
            .drop_duplicates(subset="course_name")  # 🔥 FIX
            .head(num_recs)
            .reset_index(drop=True)
    )

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
