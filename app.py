import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_resource
def load_assets():
    try:
        courses = pickle.load(open('courses.pkl', 'rb'))
        logic = pickle.load(open('model_logic.pkl', 'rb'))
        return courses, logic
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

courses, logic = load_assets()

st.title("🎓 Collaborative Course Recommender")

if courses is not None:
    # 1. Get User Input
    user_input = st.text_input("Enter User ID (e.g., 15796):", value="15796")
    
    if st.button("Generate Recommendations"):
        # 2. Fix the "Same Output" Problem: Data Type Alignment
        # We try to find the user as an Integer first, then as a String
        user_id_int = int(user_input) if user_input.isdigit() else user_input
        
        user_bias_dict = logic['user_bias']
        user_history = logic['user_history']
        
        # Check if user exists in our model
        if user_id_int in user_bias_dict:
            u_bias = user_bias_dict[user_id_int]
            history = user_history.get(user_id_int, [])
            st.success(f"✅ User {user_id_int} found! Individual Bias: {u_bias:.4f}")
        elif str(user_id_int) in user_bias_dict: # Check for string version
            u_id_str = str(user_id_int)
            u_bias = user_bias_dict[u_id_str]
            history = user_history.get(u_id_str, [])
            st.success(f"✅ User {u_id_str} found! Individual Bias: {u_bias:.4f}")
        else:
            u_bias = 0
            history = []
            st.warning("⚠️ User ID not in training data. Showing global best picks.")

        # 3. Collaborative Filtering Logic: Score = Global Mean + User Bias + Item Bias
        available = courses[~courses['course_id'].isin(history)].copy()
        
        available['recommendation_score'] = available['course_id'].apply(
            lambda x: logic['global_mean'] + u_bias + logic['item_bias'].get(x, 0)
        )
        
        # 4. Format and Display
        final_table = available[[
            'course_id', 
            'recommendation_score', 
            'course_name', 
            'instructor', 
            'rating'
        ]].sort_values(by='recommendation_score', ascending=False).head(5)
        
        final_table = final_table.reset_index(drop=True)
        
        st.subheader(f"Top Recommendations for User {user_input}")
        st.table(final_table)
