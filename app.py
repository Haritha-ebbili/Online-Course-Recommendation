import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

st.title("🎓 Online Course Recommendation System")
st.write("Content-based recommendations using course metadata")

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

st.write("📊 Columns detected:")
st.write(list(df.columns))

# -------------------------------------------------
# Detect Course Title Column
# -------------------------------------------------
title_candidates = ["course_title", "course_name", "title", "course"]

course_title_col = next(
    (c for c in title_candidates if c in df.columns),
    None
)

if course_title_col is None:
    st.error("❌ No course title column found.")
    st.stop()

df = df.rename(columns={course_title_col: "course_title"})

# -------------------------------------------------
# BUILD DESCRIPTION AUTOMATICALLY
# -------------------------------------------------
text_columns = [
    col for col in df.columns
    if col != "course_title" and df[col].dtype == "object"
]

if not text_columns:
    st.error("❌ No text columns available to build descriptions.")
    st.stop()

df["description"] = df[text_columns].fillna("").agg(" ".join, axis=1)

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
def recommend(course_name, n=5):
    idx = df[df["course_title"] == course_name].index[0]
    similarity = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    indices = similarity.argsort()[::-1][1:n+1]
    return df.iloc[indices]["course_title"]

# -------------------------------------------------
# UI
# -------------------------------------------------
selected_course = st.selectbox(
    "Select a course:",
    df["course_title"].unique()
)

num_recs = st.slider("Number of recommendations", 1, 10, 5)

if st.button("🚀 Recommend"):
    results = recommend(selected_course, num_recs)
    st.subheader("📌 Recommended Courses")
    for i, c in enumerate(results, 1):
        st.write(f"**{i}. {c}**")

st.markdown("---")
st.caption("Robust Content-Based Recommendation | No Pickle | Streamlit")
