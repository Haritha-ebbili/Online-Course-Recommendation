import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------
# Page Configuration
# ------------------------------------------------------
st.set_page_config(page_title="Online Course Recommendation", layout="wide")

st.title("🎓 Online Course Recommendation System")
st.write("Model building + recommendation executed inside Streamlit")

# ------------------------------------------------------
# Load Dataset (same as notebook)
# ------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("online_course_recommendation.xlsx")
    return df

df = load_data()

# ------------------------------------------------------
# Data Preprocessing (same as EDA notebook)
# ------------------------------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Identify course title column
title_cols = ["course_title", "course_name", "title", "course"]
title_col = next((c for c in title_cols if c in df.columns), None)

if title_col is None:
    st.error("No course title column found in dataset.")
    st.stop()

df.rename(columns={title_col: "course_title"}, inplace=True)

# Build text feature (same logic as model building)
text_columns = [
    col for col in df.columns
    if col != "course_title" and df[col].dtype == "object"
]

df[text_columns] = df[text_columns].fillna("")

df["combined_text"] = df[text_columns].agg(" ".join, axis=1)

# ------------------------------------------------------
# Model Building (TF-IDF) – SAME AS NOTEBOOK
# ------------------------------------------------------
st.subheader("⚙️ Model Building")

@st.cache_resource
def train_model(text):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(text)
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = train_model(df["combined_text"])

st.success("TF-IDF model trained successfully")

# ------------------------------------------------------
# Similarity Computation
# ------------------------------------------------------
@st.cache_resource
def compute_similarity(matrix):
    return cosine_similarity(matrix)

cosine_sim = compute_similarity(tfidf_matrix)

# ------------------------------------------------------
# Recommendation Function (Notebook Logic)
# ------------------------------------------------------
def recommend_courses(course_name, n=5):
    idx = df[df["course_title"] == course_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n+1]

    course_indices = [i[0] for i in scores]
    return df.iloc[course_indices]["course_title"]

# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.subheader("🔍 Course Recommendation")

selected_course = st.selectbox(
    "Select a course:",
    df["course_title"].unique()
)

num_recommendations = st.slider(
    "Number of recommendations",
    1, 10, 5
)

if st.button("🚀 Recommend Courses"):
    results = recommend_courses(selected_course, num_recommendations)

    st.subheader("📌 Recommended Courses")
    for i, course in enumerate(results, 1):
        st.write(f"**{i}. {course}**")

# ------------------------------------------------------
# Footer
# ------------------------------------------------------
st.markdown("---")
st.caption("Model built & executed live using TF-IDF + Cosine Similarity (No Pickle)")

