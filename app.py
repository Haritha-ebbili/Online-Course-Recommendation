import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

st.title("🎓 Online Course Recommendation System")
st.write("Content-based recommendation using TF-IDF and cosine similarity")

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("online_course_recommendation.xlsx")

df = load_data()

# ----------------------------------------------------
# Data Preprocessing (Model Building Step)
# ----------------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Identify course title column
title_candidates = ["course_title", "course_name", "title", "course"]
title_col = next((c for c in title_candidates if c in df.columns), None)

if title_col is None:
    st.error("Dataset must contain a course title column.")
    st.stop()

df.rename(columns={title_col: "course_title"}, inplace=True)

# Combine all text-based features
text_columns = [
    col for col in df.columns
    if col != "course_title" and df[col].dtype == "object"
]

if not text_columns:
    st.error("No text features found for model building.")
    st.stop()

df[text_columns] = df[text_columns].fillna("")
df["combined_text"] = df[text_columns].agg(" ".join, axis=1)

# ----------------------------------------------------
# Model Building (TF-IDF)
# ----------------------------------------------------
@st.cache_resource
def build_model(text_data):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix

tfidf_matrix = build_model(df["combined_text"])

st.success("Model built successfully using TF-IDF")

# ----------------------------------------------------
# Recommendation Logic (Final Model Output)
# ----------------------------------------------------
def recommend_courses(course_name, top_n=5):
    idx = df[df["course_title"] == course_name].index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
    return df.iloc[similar_indices]["course_title"]

# ----------------------------------------------------
# User Interface
# ----------------------------------------------------
st.subheader("🔍 Select a Course")

selected_course = st.selectbox(
    "Choose a course:",
    df["course_title"].unique()
)

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("🚀 Recommend Courses"):
    recommendations = recommend_courses(
        selected_course, num_recommendations
    )

    st.subheader("📌 Recommended Courses")
    for i, course in enumerate(recommendations, 1):
        st.write(f"**{i}. {course}**")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.caption("Final Deployed Model: TF-IDF + Cosine Similarity (Content-Based)")
