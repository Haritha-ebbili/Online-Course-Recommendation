#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hybrid Course Recommender", layout="wide")

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_assets():
    courses = pickle.load(open("courses.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    logic = pickle.load(open("model_logic.pkl", "rb"))
    return courses, similarity, logic

courses, similarity, logic = load_assets()

# ---------------- UI ----------------
st.title("🎓 Hybrid Course Recommendation System")
st.caption("Collaborative + Content-Based Filtering")

user_input = st.text_input("Enter User ID", value="15796")

# ---------------- RECOMMENDATION ----------------
if st.button("Generate Recommendations"):

    user_id = int(user_input) if user_input.isdigit() else user_input

    user_bias = logic["user_bias"].get(user_id, 0)
    history = logic["user_history"].get(user_id, [])

    # Remove watched courses
    available = courses[~courses["course_id"].isin(history)].copy()

    # User preference signal
    if history:
        user_mean = courses[courses["course_id"].isin(history)]["rating"].mean()
    else:
        user_mean = courses["rating"].mean()

    # Collaborative score
    available["cf_score"] = available["course_id"].apply(
        lambda x: logic["global_mean"] + user_bias + logic["item_bias"].get(x, 0)
    )

    # Content-based score (if history exists)
    if history:
        idx_map = {cid: idx for idx, cid in enumerate(courses["course_id"])}
        sim_scores = []

        for cid in available["course_id"]:
            scores = []
            for h in history:
                if h in idx_map:
                    scores.append(similarity[idx_map[cid]][idx_map[h]])
            sim_scores.append(np.mean(scores) if scores else 0)

        available["content_score"] = sim_scores
    else:
        available["content_score"] = 0

    # Final Hybrid Score
    available["final_score"] = (
        available["cf_score"]
        + 0.5 * available["content_score"]
        + 0.3 * (available["rating"] - user_mean)
    )

    # Top 5
    top_courses = (
        available
        .sort_values("final_score", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    # Display
    st.subheader("📌 Recommended Courses")
    st.dataframe(
        top_courses[
            ["course_id", "course_name", "instructor", "rating", "final_score"]
        ],
        use_container_width=True
    )


# In[ ]:




