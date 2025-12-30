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
    "An **advanced content-based recommendation system** built using "
    "**TF-IDF vectorization** and **cosine similarity**, deployed with Streamlit."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("online_course_recommendation.xlsx")

df = load_data()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"Total Courses: {df.shape[0]} | Total Features: {df.shape[1]}")

# --------------------------------------------------
# Data Preprocessing (Model Building)
# --------------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Detect course title column
title_candidates = ["course_title", "course_name", "title", "course"]
title_col = next((c for c in title_candidates if c in df.columns), None)

if title_col is None:
    st.error("Dataset must contain a course title column.")
    st.stop()

df.rename(columns={title_col: "course_title"}, inplace=True)

# Combine all textual features
text_columns = [
    col for col in df.columns
    if col != "course_title" and df[col].dtype == "object"
]

if not text_columns:
    st.error("No textual features available for model building.")
    st.stop()

df[text_columns] = df[text_columns].fillna("")
df["combined_text"] = df[text_columns].agg(" ".join, axis=1)

# --------------------------------------------------
# Model Building (TF-IDF)
# --------------------------------------------------
@st.cache_resource
def build_tfidf_model(text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(text)
    return tfidf_matrix

tfidf_matrix = build_tfidf_model(df["combined_text"])
st.success("✅ TF-IDF model successfully built")

# --------------------------------------------------
# Recommendation Logic (with similarity scores)
# --------------------------------------------------
def recommend_courses(course_name, top_n=5, threshold=0.6):
    idx = df[df["course_title"] == course_name].index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    scored = list(enumerate(similarity_scores))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in scored[1:]:
        if score >= threshold:
            results.append((df.iloc[i]["course_title"], score))
        if len(results) == top_n:
            break

    return pd.DataFrame(results, columns=["Course Title", "Similarity Score"])

# --------------------------------------------------
# User Controls
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

similarity_threshold = st.slider(
    "Minimum similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05
)

# --------------------------------------------------
# Recommendation Output
# --------------------------------------------------
if st.button("🚀 Recommend Courses"):
    results = recommend_courses(
        selected_course,
        num_recommendations,
        similarity_threshold
    )

    if results.empty:
        st.warning("No courses meet the selected similarity threshold.")
    else:
        st.subheader("📌 Recommended Courses")

        for _, row in results.iterrows():
            st.markdown(f"**{row['Course Title']}**")
            st.progress(float(row["Similarity Score"]))
            st.caption(f"Similarity Score: {row['Similarity Score']:.3f}")

        # --------------------------------------------------
        # Similarity Distribution Chart
        # --------------------------------------------------
        st.subheader("📈 Similarity Score Distribution")
        st.bar_chart(
            results.set_index("Course Title")["Similarity Score"]
        )

# --------------------------------------------------
# Final Model Summary
# --------------------------------------------------
st.markdown("---")
st.subheader("📘 Final Deployed Model Summary")

st.markdown(
    """
**Model Type:** Content-Based Recommendation System  

**Feature Engineering:**  
- All textual attributes combined into a unified feature space  

**Vectorization:**  
- TF-IDF (Term Frequency–Inverse Document Frequency)  

**Similarity Measure:**  
- Cosine Similarity  

**Model Strengths:**  
- No dependency on user history  
- Explainable similarity scores  
- Scales efficiently for large datasets  
- Threshold-based quality control  

**Deployment:**  
- Model built dynamically within Streamlit  
- No serialized (pickle) files  
- Stable, reproducible, and cloud-safe  
"""
)

st.caption("Final Deployed Model: TF-IDF + Cosine Similarity (Content-Based)")
