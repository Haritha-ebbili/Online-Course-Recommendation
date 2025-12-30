import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Online Course Recommendation", layout="wide")
st.title("🎓 Online Course Recommendation System")

@st.cache_data
def load_data():
    return pd.read_excel("online_course_recommendation.xlsx")

df = load_data()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

title_candidates = ["course_title", "course_name", "title", "course"]
title_col = next((c for c in title_candidates if c in df.columns), None)

if title_col is None:
    st.error("No course title column found.")
    st.stop()

df.rename(columns={title_col: "course_title"}, inplace=True)

text_cols = [c for c in df.columns if c != "course_title" and df[c].dtype == "object"]
df[text_cols] = df[text_cols].fillna("")
df["combined_text"] = df[text_cols].agg(" ".join, axis=1)

@st.cache_resource
def build_tfidf(text):
    return TfidfVectorizer(stop_words="english").fit_transform(text)

tfidf_matrix = build_tfidf(df["combined_text"])

def recommend_courses(course_name, n=5):
    idx = df[df["course_title"] == course_name].index[0]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    indices = scores.argsort()[::-1][1:n+1]
    return df.iloc[indices]["course_title"]

selected_course = st.selectbox("Choose a course:", df["course_title"].unique())
num_recs = st.slider("Number of recommendations", 1, 10, 5)

if st.button("🚀 Recommend"):
    for i, c in enumerate(recommend_courses(selected_course, num_recs), 1):
        st.write(f"**{i}. {c}**")

st.markdown("---")
st.caption("TF-IDF + Cosine Similarity | No Pickle | No Cache Errors")
