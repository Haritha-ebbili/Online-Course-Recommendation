import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(page_title="Online Course Recommendation System", layout="wide")
st.title("🎓 Online Course Recommendation System")
st.write("Model building executed live (TF-IDF + Cosine Similarity)")

# ------------------------------------------------------
# Load Dataset (safe to cache)
# ------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("online_course_recommendation.xlsx")

df = load_data()

# ------------------------------------------------------
# Normalize Column Names
# ------------------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ------------------------------------------------------
# Detect Course Title Column
# ------------------------------------------------------
title_candidates = ["course_title", "course_name", "title", "course"]
title_col = next((c for c in title_candidates if c in df.columns), None)

if title_col is None:
    st.error("No course title column found.")
    st.write("Columns found:", list(df.columns))
    st.stop()

df.rename(columns={title_col: "course_title"}, inplace=True)

# ------------------------------------------------------
# Build Text Feature (model-building logic)
# ------------------------------------------------------
text_cols = [
    c for c in df.columns
    if c != "course_title" and df[c].dtype == "object"
]

if not text_cols:
    st.error("No text columns available for model building.")
    st.stop()

df[text_cols] = df[text_cols].fillna("")
df["combined_text"] = df[text_cols].agg(" ".join, axis=1)

# ------------------------------------------------------
# TF-IDF MODEL (safe to cache)
# ------------------------------------------------------
@st.cache_resource
def build_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(text)
    return matrix

tfidf_matrix = build_tfidf(df["combined_text"])

st.success("TF-IDF model built successfully")

# ------------------------------------------------------
# Recommendation Function (NO CACHE HERE)
# ------------------------------------------------------
def recommend_courses(course_name, n=5):
    idx = df[df["course_title"] == course_name].index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:n+1]
    return df.iloc[similar_indices]["course_title"]

# ------------------------------------------------------
# UI
# ------------------------------------------------------
st.subheader("🔍 Select a Course")

selected_course = st.selectbox(
    "Choose a course:",
    df["course_title"].unique()
)

num_recommendations = st.slider("Number of recommendations", 1, 10, 5)

if st.button("🚀 Recommend"):
    results = recommend_courses(selected_course, num_recommendations)

    st.subheader("📌 Recommended Courses")
    for i, course in enumerate(results, 1):
        st.write(f"**{i}. {course}**")

st.markdown("---")
st.caption("TF-IDF + Cosine Similarity | No Pickle | No Cache Errors")
