import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

st.title("🎓 Online Course Recommendation System")
st.write(
    "Deployment of the **content-based recommendation model** built using "
    "TF-IDF vectorization and cosine similarity."
)

# --------------------------------------------------
# Load Dataset (Same as Notebook)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("online_course_recommendation.xlsx")

df = load_data()

# --------------------------------------------------
# DATASET TABLE (AS REQUESTED)
# --------------------------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df, use_container_width=True)
st.caption(
    f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}"
)

# --------------------------------------------------
# Data Preprocessing (Exact Model Building Logic)
# --------------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Detect course title column
title_candidates = ["course_title", "course_name", "title", "course"]
title_col = next((c for c in title_candidates if c in df.columns), None)

if title_col is None:
    st.error("Dataset must contain a course title column.")
    st.stop()

df.rename(columns={title_col: "course_title"}, inplace=True)

# Combine all text columns (same approach used in notebook)
text_columns = [
    col for col in df.columns
    if col != "course_title" and df[col].dtype == "object"
]

df[text_columns] = df[text_columns].fillna("")
df["combined_text"] = df[text_columns].agg(" ".join, axis=1)

# --------------------------------------------------
# MODEL BUILDING (TF-IDF)
# --------------------------------------------------
@st.cache_resource
def build_model(text_data):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix

tfidf_matrix = build_model(df["combined_text"])

st.success("✅ Model built successfully using TF-IDF")

# --------------------------------------------------
# Recommendation Function (Exact Output of Model)
# --------------------------------------------------
def recommend_courses(course_name, top_n=5):
    idx = df[df["course_title"] == course_name].index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
    return df.iloc[similar_indices]["course_title"]

# --------------------------------------------------
# User Interface (Deployment Layer)
# --------------------------------------------------
st.subheader("🔍 Course Recommendation")

selected_course = st.selectbox(
    "Select a course:",
    df["course_title"].unique()
)

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("🚀 Recommend"):
    results = recommend_courses(selected_course, num_recommendations)

    st.subheader("📌 Recommended Courses")
    for i, course in enumerate(results, 1):
        st.write(f"{i}. {course}")

# --------------------------------------------------
# Final Model Statement
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Final Deployed Model: TF-IDF + Cosine Similarity (Content-Based Recommendation)"
)
