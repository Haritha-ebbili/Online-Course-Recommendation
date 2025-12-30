import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

st.title("🎓 Online Course Recommendation System")
st.write("Content-based course recommendation using TF-IDF")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("online_course_recommendation.xlsx")

df = load_data()

# -------------------------------------------------
# Normalize Column Names
# -------------------------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# -------------------------------------------------
# DEBUG: Show columns (helps deployment)
# -------------------------------------------------
st.write("📊 Dataset Columns Detected:")
st.write(list(df.columns))

# -------------------------------------------------
# AUTO COLUMN MAPPING
# -------------------------------------------------
title_candidates = [
    "course_title", "course_name", "title", "course"
]

description_candidates = [
    "description", "course_description", "overview",
    "about_course", "summary", "details"
]

course_title_col = None
description_col = None

for col in title_candidates:
    if col in df.columns:
        course_title_col = col
        break

for col in description_candidates:
    if col in df.columns:
        description_col = col
        break

# -------------------------------------------------
# Validate Columns
# -------------------------------------------------
if course_title_col is None or description_col is None:
    st.error("❌ Required columns not found automatically.")
    st.write("Available columns:", list(df.columns))
    st.stop()

# -------------------------------------------------
# Rename for internal consistency
# -------------------------------------------------
df = df.rename(columns={
    course_title_col: "course_title",
    description_col: "description"
})

# -------------------------------------------------
# Handle Missing Values
# -------------------------------------------------
df["description"] = df["description"].fillna("")

# -------------------------------------------------
# TF-IDF Vectorization
# -------------------------------------------------
@st.cache_resource
def build_tfidf(text):
    tfidf = TfidfVectorizer(stop_words="english")
    return tfidf.fit_transform(text)

tfidf_matrix = build_tfidf(df["description"])

# -------------------------------------------------
# Recommendation Function
# -------------------------------------------------
def recommend(course_name, top_n=5):
    idx = df[df["course_title"] == course_name].index[0]

    similarity = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    indices = similarity.argsort()[::-1][1:top_n + 1]
    return df.iloc[indices]["course_title"]

# -------------------------------------------------
# UI
# -------------------------------------------------
st.subheader("🔍 Select a Course")

selected_course = st.selectbox(
    "Choose a course:",
    df["course_title"].unique()
)

num_recommendations = st.slider(
    "Number of recommendations",
    1, 10, 5
)

if st.button("🚀 Recommend Courses"):
    results = recommend(selected_course, num_recommendations)

    st.subheader("📌 Recommended Courses")
    for i, course in enumerate(results, 1):
        st.write(f"**{i}. {course}**")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Streamlit + TF-IDF | Robust Column Mapping | No Pickle")
