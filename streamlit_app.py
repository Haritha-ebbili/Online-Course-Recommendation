#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pickle
import pandas as pd
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Course Recommender", layout="wide")
st.title("🎓 Collaborative Course Recommender")

# --------------------------------------------------
# Load Assets (NO CACHE – SAFE)
# --------------------------------------------------
def load_assets():
    try:
        if not os.path.exists("courses.pkl"):
            st.error("❌ courses.pkl not found in root directory")
            return None, None

        if not os.path.exists("model_logic.pkl"):
            st.error("❌ model_logic.pkl not found in root directory")
            return None, None

        with open("courses.pkl", "rb") as f:
            courses = pickle.load(f)

        with open("model_logic.pkl", "rb") as f:
            logic = pickle.load(f)

        return courses, logic

    except Exception as e:
        st.error(f"❌ Error loading pickle files: {e}")
        return None, None


courses, logic = load_assets()

# --------------------------------------------------
# Stop app if files failed to load
# --------------------------------------------------
if courses is None or logic is None:
    st.stop()

# --------------------------------------------------
# User Input
# --------------------------------------------------
user_input = st.text_input("Enter User ID (e.g., 15796):", value="15796")

if st.button("Generate Recommendations"):

    # --------------------------------------------------
    # Handle User ID Robustly
    # --------------------------------------------------
    user_id = int(user_input) if user_input.isdigit() else str(user_input)

    user_bias_dict = logic.get("user_bias", {})
    user_history = logic.get("user_history", {})
    item_bias = logic.get("item_bias", {})
    global_mean = logic.get("global_mean", 0)

    # --------------------------------------------------
    # Fetch User Bias & History
    # --------------------------------------------------
    if user_id in user_bias_dict:
        u_bias = user_bias_dict[user_id]
        history = user_history.get(user_id, [])
        st.success(f"✅ User {user_id} found | Bias = {u_bias:.4f}")

    elif str(user_id) in user_bias_dict:
        u_bias = user_bias_dict[str(user_id)]
        history = user_history.get(str(user_id), [])
        st.success(f"✅ User {user_id} found | Bias = {u_bias:.4f}")

    else:
        u_bias = 0
        history = []
        st.warning("⚠️ User not in training data. Showing global recommendations.")

    # --------------------------------------------------
    # Collaborative Filtering Scoring
    # Score = global_mean + user_bias + item_bias
    # --------------------------------------------------
    available = courses[~courses["course_id"].isin(history)].copy()

    available["recommendation_score"] = available["course_id"].apply(
        lambda x: global_mean + u_bias + item_bias.get(x, 0)
    )

    # --------------------------------------------------
    # Final Output
    # --------------------------------------------------
    final_table = (
        available[
            ["course_id", "recommendation_score", "course_name", "instructor", "rating"]
        ]
        .sort_values(by="recommendation_score", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    st.subheader(f"🏆 Top Recommendations for User {user_id}")
    st.dataframe(final_table, use_container_width=True)
