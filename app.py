import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Pro Course Recommender", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp { background-color: #D5CABD !important; }
.main-header {
    font-size: 3.5rem !important;
    background: linear-gradient(90deg, #6a1b9a, #ec407a) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
}
.stButton > button {
    background: linear-gradient(45deg, #1e3c72, #7b1fa2) !important;
    color: white !important;
    border-radius: 50px !important;
    padding: 15px 40px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= DATA ENGINE =================
@st.cache_data
def load_data():
    # Loading your specific dataset
    df = pd.read_pickle("full_data.pkl")
    return df

df = load_data()

def get_recommendations(target_user_id, n_recs):
    # 1. Create User-Item Matrix (Rows=Users, Cols=Courses, Values=Ratings)
    # We use pivot_table to handle potential duplicate user-course entries
    user_item_matrix = df.pivot_table(index='user_id', columns='course_name', values='rating').fillna(0)
    
    if target_user_id not in user_item_matrix.index:
        return pd.DataFrame(), []

    # 2. Calculate Cosine Similarity between the target user and all others
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # 3. Get Top 5 similar users
    similar_users = user_sim_df[target_user_id].sort_values(ascending=False)[1:6].index
    
    # 4. Get courses taken by similar users but NOT by the target user
    target_user_courses = df[df['user_id'] == target_user_id]['course_name'].unique()
    
    # Filter data for similar users and exclude target user's courses
    candidate_df = df[df['user_id'].isin(similar_users) & ~df['course_name'].isin(target_user_courses)]
    
    # 5. Aggregate and ensure UNIQUE course names
    # We group by course_name to get a single entry per course
    rec_result = candidate_df.groupby('course_name').agg({
        'course_id': 'first',
        'instructor': 'first',
        'rating': 'mean'
    }).reset_index()
    
    # Sort by rating and limit
    rec_result = rec_result.sort_values(by='rating', ascending=False).head(n_recs)
    rec_result['recommendation_score'] = rec_result['rating']
    
    return rec_result, target_user_courses

# ================= UI LAYOUT =================
st.markdown('<h1 class="main-header">Course Recommendation System</h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.header("Step 1: User Profile")
    user_id = st.number_input("Enter User ID", min_value=1, value=15796)
with col2:
    st.header("Step 2: Preference")
    num_recommendations = st.slider("Number of unique courses?", 1, 20, 10)

# ================= EXECUTION =================
if st.button("Generate Personalised Recommendations"):
    recommendations, history = get_recommendations(user_id, num_recommendations)
    
    if not recommendations.empty:
        st.session_state.recommendations = recommendations
        st.session_state.course_options = recommendations["course_name"].tolist()
        st.success(f"Found {len(recommendations)} new courses based on similar learners!")
    else:
        st.error("User ID not found or no similar profiles available.")

# ================= DISPLAY =================
if "recommendations" in st.session_state:
    st.header("Step 3: Recommended for You")
    st.dataframe(
        st.session_state.recommendations[["course_id", "course_name", "instructor", "rating"]].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.header("Step 4: Select to Enroll")
    selected_courses = st.multiselect("Choose courses:", st.session_state.course_options)

    if selected_courses:
        st.header("Step 5: Final Selection Details")
        final_df = st.session_state.recommendations[st.session_state.recommendations["course_name"].isin(selected_courses)]
        st.table(final_df[["course_name", "instructor", "rating"]])
