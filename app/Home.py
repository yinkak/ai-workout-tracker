
import streamlit as st
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration for the entire app ---
st.set_page_config(
    page_title="Workout AI Coach",
    page_icon="🏋️",
    layout="centered",
    initial_sidebar_state="expanded" # Keep sidebar expanded for easy navigation
)

# Ensure necessary directories exist (good to keep this)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Home Page Content ---
# Streamlit will automatically create the sidebar navigation from files in the 'pages' folder.
# The content of app.py is what displays when no specific page is selected (i.e., the home page).

st.title("Welcome to Your Personal AI Workout Coach! 🏋️")
st.markdown("""
Your smart companion for optimizing your strength training. This app leverages your past workout data
to provide personalized recommendations for your next session, helping you achieve progressive overload efficiently and safely.
""")

st.image("https://via.placeholder.com/600x300?text=Your+Workout+Coach+Image+Here", caption="Your journey to stronger, smarter training starts here!", use_column_width=True)
# Replace the placeholder URL with an actual image related to workouts/AI.

st.subheader("What can you do here?")

st.markdown("""
Navigate using the sidebar on the left to:
* **➕ Log Workouts:** Record your completed sets, reps, and RPE to build your training history.
* **💪 Get Recommendation:** Receive AI-powered suggestions for your next exercise weight.
* **📚 View History:** Browse and analyze your past training sessions to track your progress.
* **🧠 Model Insights:** Understand how the AI coach works and what factors influence its recommendations.
""")

st.markdown("---")
st.info("Start by logging your first workout, or get a recommendation based on your existing data!")

st.sidebar.markdown("---")
st.sidebar.caption("App Version 1.0")

# No more manual page switching logic here! Streamlit handles it automatically.