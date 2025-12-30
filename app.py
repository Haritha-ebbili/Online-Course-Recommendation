#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Online Course Recommendation", page_icon="🎓")

@st.cache_data
def load_data():
    df = pd.read_excel("online_course_recommendation.xlsx")
    df.fillna("", inplace=True)
    df.rename(columns={
        "course_name": "course_title",
        "difficulty_level": "level"
    }, inplace=True)
    return df.reset_index(drop=True)

@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    return tfidf, tfidf_matrix

courses = load_data()
tfidf, tfidf_matrix = load_models()

def recommend(course_title, n=5):
    idx = courses[courses["course_title"] == course_title].index[0]
    query_vec = tfidf_matrix[idx]
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[::-1][1:n+1]
    return courses.iloc[top_indices]

st.title("🎓 Online Course Recommendation System")

selected_course = st.selectbox(
    "Select a course",
    sorted(courses["course_title"].unique())
)

num = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend"):
    results = recommend(selected_course, num)
    for _, row in results.iterrows():
        st.markdown(f"""
        **📘 {row['course_title']}**  
        👨‍🏫 Instructor: {row['instructor']}  
        🎯 Level: {row['level']}  
        ⭐ Rating: {row['rating']}  
        ---
        """)


# In[ ]:




