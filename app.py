import streamlit as st
import pandas as pd

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

st.title("🎓 Online Course Recommendation System")
st.write(
    "Deployment of the **final popularity-based recommendation model**, "
    "selected based on **lowest RMSE performance**."
)

# --------------------------------------------------
# Load Final Model Output (Result of Model Building)
# --------------------------------------------------
@st.cache_data
def load_recommendations():
    return pd.read_csv("final_recommendations.csv")

df = load_recommendations()

# --------------------------------------------------
# DATASET PREVIEW (MODEL OUTPUT TABLE)
# --------------------------------------------------
st.subheader("📊 Final Model Output – Dataset Preview")

st.dataframe(
    df,
    use_container_width=True
)

st.caption(
    f"Total Recommended Courses: {df.shape[0]}"
)

# --------------------------------------------------
# Model Description
# --------------------------------------------------
st.markdown(
    """
### 📌 Model Selected
- **Model Type:** Popularity-Based Recommendation  
- **Selection Criteria:** Lowest RMSE  
- **Recommendation Logic:**  
  Courses are ranked based on a computed popularity score derived from
  historical engagement and ratings.
"""
)

# --------------------------------------------------
# User Controls
# --------------------------------------------------
st.subheader("🔍 Top Course Recommendations")

top_n = st.slider(
    "Select number of top recommendations to display",
    min_value=1,
    max_value=len(df),
    value=5
)

# --------------------------------------------------
# Recommendation Display
# --------------------------------------------------
top_courses = (
    df.sort_values(by="recommendation_score", ascending=False)
      .head(top_n)
)

st.subheader("🏆 Recommended Courses")

st.dataframe(
    top_courses[[
        "course_name",
        "instructor",
        "rating",
        "recommendation_score"
    ]],
    use_container_width=True
)

# --------------------------------------------------
# Final Model Conclusion
# --------------------------------------------------
st.markdown("---")
st.subheader("📘 Final Deployed Model Summary")

st.markdown(
    """
**Final Model:** Popularity-Based Recommendation System  

**Why This Model?**
- Achieved the **lowest RMSE** among evaluated models
- Stable and robust for real-world deployment
- Does not require user history at inference time

**Output:**
- Ranked list of top courses
- Includes instructor, rating, and popularity score

**Deployment:**
- Model results are precomputed during training
- Streamlit app serves the final recommendations efficiently
"""
)

st.caption(
    "Final Deployed Model: Popularity-Based Recommendation (Lowest RMSE)"
)
