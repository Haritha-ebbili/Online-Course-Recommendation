import streamlit as st
import pandas as pd
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Personalized Course Recommender", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp { background-color: #D5CABD !important; }
.main-header {
    font-size: 3rem !important;
    background: linear-gradient(90deg, #1e3c72, #7b1fa2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    text-align: center;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(45deg, #1e3c72, #7b1fa2) !important;
    color: white !important;
    border-radius: 10px !important;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= DATA LOADING =================
@st.cache_data
def load_data():
    if os.path.exists("full_data.pkl"):
        return pd.read_pickle("full_data.pkl")
    else:
        # Fallback/Sample data structure if file not found
        cols = ['user_id', 'course_id', 'course_name', 'instructor', 'rating']
        return pd.DataFrame(columns=cols)

df = load_data()

# ================= SESSION STATE INITIALIZATION =================
# This prevents the UI from disappearing when you interact with widgets
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None

# ================= UI HEADER =================
st.markdown('<h1 class="main-header">Course Recommendation System</h1>', unsafe_allow_html=True)

# STEPS 1 & 2: INPUTS
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Enter User ID")
    user_id = st.number_input("User ID", min_value=1, value=15796, step=1)

with col2:
    st.header("Step 2: Recommendations")
    num_recs = st.slider("Number of unique courses?", 1, 20, 10)

# RECOMMENDATION LOGIC
if st.button("Generate Recommendations"):
    # 1. Identify user history
    user_history = df[df["user_id"] == user_id]["course_name"].unique()
    
    # 2. Collaborative Filtering: Find users who took the same courses
    similar_users = df[df["course_name"].isin(user_history) & (df["user_id"] != user_id)]["user_id"].unique()
    
    # 3. Get candidate courses (taken by similar users, not by target user)
    candidates = df[df["user_id"].isin(similar_users) & ~df["course_name"].isin(user_history)]
    
    # Fallback: If no similar users, suggest top rated courses not taken by user
    if candidates.empty:
        candidates = df[~df["course_name"].isin(user_history)]

    # 4. Aggregating to ensure UNIQUE course names
    unique_recs = candidates.groupby("course_name").agg({
        "course_id": "first",
        "instructor": "first",
        "rating": "mean"
    }).reset_index()

    # 5. Sort and Save
    unique_recs = unique_recs.sort_values(by="rating", ascending=False).head(num_recs)
    unique_recs["recommendation_score"] = unique_recs["rating"].round(2)
    
    st.session_state.recommendations = unique_recs
    st.session_state.selected_user = user_id

# ================= OUTPUT STEPS 3, 4, 5 =================

if st.session_state.recommendations is not None:
    st.divider()
    
    # STEP 3: DISPLAY RECOMMENDATIONS
    st.header(f"Step 3: Recommended for User {st.session_state.selected_user}")
    rec_display = st.session_state.recommendations[["course_id", "recommendation_score", "course_name", "instructor", "rating"]]
    st.dataframe(rec_display, use_container_width=True, hide_index=True)

    # STEP 4: SELECTION
    st.header("Step 4: Select Courses")
    all_options = st.session_state.recommendations["course_name"].tolist()
    selected_names = st.multiselect("Pick courses to see details:", options=all_options)

    # STEP 5: FILTERED RANKING (4 to 5)
    if selected_names:
        st.header("Step 5: Selected Courses (Highly Rated 4.0 - 5.0)")
        
        # Filter logic: Must be in selection AND rating between 4 and 5
        step5_df = df[
            (df["course_name"].isin(selected_names)) & 
            (df["rating"] >= 4.0) & 
            (df["rating"] <= 5.0)
        ].copy()
        
        # Ensure unique course names in the final display
        step5_df = step5_df.drop_duplicates(subset="course_name")
        
        if not step5_df.empty:
            # Add a ranking column based on rating
            step5_df = step5_df.sort_values(by="rating", ascending=False)
            st.table(step5_df[["course_id", "course_name", "instructor", "rating"]])
        else:
            st.warning("None of the selected courses meet the 4.0 - 5.0 rating criteria.")
    else:
        st.info("Please select courses in Step 4 to proceed.")
