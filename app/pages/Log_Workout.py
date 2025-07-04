import streamlit as st
import pandas as pd
from datetime import datetime
import os
import subprocess # To run shell commands like python transform.py

# --- File Paths (adjust as needed) ---
RAW_WORKOUT_LOG_PATH = "../data/50-Day_workout_log.csv"
TRANSFORM_SCRIPT_PATH = "../src/transform.py"
TRAIN_SCRIPT_PATH = "../src/train_model.py"

# --- Function to log workout ---
def log_workout(date, exercise, weight_lbs, reps, sets, rpe, notes):
    new_entry = pd.DataFrame([{
        'date': date.strftime('%Y-%m-%d'), # Format date to string for CSV
        'exercise': exercise,
        'weight_lbs': weight_lbs,
        'reps': reps,
        'sets': sets,
        'rpe': rpe,
        'notes': notes
    }])

    # Check if raw_workout_log.csv exists. If not, create with header.
    if not os.path.exists(RAW_WORKOUT_LOG_PATH):
        new_entry.to_csv(RAW_WORKOUT_LOG_PATH, mode='w', index=False, header=True)
    else:
        new_entry.to_csv(RAW_WORKOUT_LOG_PATH, mode='a', index=False, header=False)
    
    st.success("Workout logged successfully!")
    st.markdown("---")
    st.write("Newly added entry:")
    st.dataframe(new_entry)

    # Trigger re-transformation and re-training
    st.info("Re-training AI Coach with your new data... This may take a moment.")
    try:
        # Run the transformation script
        subprocess.run(['python3', TRANSFORM_SCRIPT_PATH], check=True, capture_output=True)
        st.success("Data transformed!")

        # Run the training script
        subprocess.run(['python3', TRAIN_SCRIPT_PATH], check=True, capture_output=True)
        st.success("Model re-trained with new data! Your recommendations are now up-to-date.")

        # Invalidate Streamlit caches that depend on the model/data
        st.cache_data.clear()
        st.cache_resource.clear()

        st.rerun() # Rerun the app to load new model/data into cache

    except subprocess.CalledProcessError as e:
        st.error(f"Error during re-training process: {e.stderr.decode()}")
        st.error("Please check your `src/transform.py` and `src/train_model.py` scripts.")
    except Exception as e:
        st.error(f"An unexpected error occurred during re-training: {e}")

# --- Log Workout Page UI ---
def show_log_workout_page():
    st.title("➕ Log Your Workout")
    st.markdown("Enter the details of your just-completed workout to update your progress.")

    # Try to load existing exercises to pre-fill selectbox
    existing_exercises = []
    if os.path.exists(RAW_WORKOUT_LOG_PATH):
        try:
            df_raw_history = pd.read_csv(RAW_WORKOUT_LOG_PATH)
            existing_exercises = sorted(df_raw_history['exercise'].unique().tolist())
        except Exception as e:
            st.warning(f"Could not load existing exercises: {e}")
            existing_exercises = [] # Fallback to empty list

    with st.form("log_workout_form", clear_on_submit=True):
        log_date = st.date_input("Workout Date", datetime.now())

        logged_exercise = st.selectbox(
            "**Exercise:**",
            options=existing_exercises,
            index=0 if existing_exercises else None,
            accept_new_options=True,
            key="logged_exercise_select"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            logged_weight = st.number_input(
                "**Weight (lbs):**",
                min_value=0.0,
                value=50.0,
                step=2.5,
                key="logged_weight"
            )
        with col2:
            logged_reps = st.number_input(
                "**Reps:**",
                min_value=1,
                value=8,
                step=1,
                key="logged_reps"
            )
        with col3:
            logged_rpe = st.slider(
                "**RPE (1-10):**",
                min_value=1,
                max_value=10,
                value=7,
                step=1,
                key="logged_rpe"
            )
        
        logged_sets = st.number_input(
            "**Sets:**",
            min_value=1,
            value=3,
            step=1,
            key="logged_sets"
        )

        logged_notes = st.text_area(
            "**Notes (optional):**",
            placeholder="e.g., 'Felt strong today', 'Struggled on last set', 'Form breakdown'",
            key="logged_notes"
        )

        submitted = st.form_submit_button("Log Workout")

        if submitted:
            if logged_exercise:
                log_workout(log_date, logged_exercise, logged_weight, logged_reps, logged_sets, logged_rpe, logged_notes)
            else:
                st.warning("Please enter an exercise name.")

# Call the function if this script is run directly (for testing, but mainly for Streamlit)
if __name__ == '__main__':
    show_log_workout_page()