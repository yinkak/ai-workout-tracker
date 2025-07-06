# src/utils.py

import streamlit as st
import pandas as pd
import os
import subprocess
from datetime import datetime

# --- File Paths (Relative to project root, as fixed before) ---
RAW_WORKOUT_LOG_PATH = "data/raw_workout_log.csv"
TRANSFORM_SCRIPT_PATH = "src/transform.py"
TRAIN_SCRIPT_PATH = "src/train_model.py"

def log_workout(date, exercise, weight_lbs, reps, sets, rpe, notes):
    new_entry = pd.DataFrame([{
        'date': date.strftime('%Y-%m-%d'),
        'exercise': exercise,
        'weight_lbs': weight_lbs,
        'reps': reps,
        'sets': sets,
        'rpe': rpe,
        'notes': notes
    }])

    # Ensure the parent directory for the CSV exists
    os.makedirs(os.path.dirname(RAW_WORKOUT_LOG_PATH), exist_ok=True)
    
    # Append to CSV
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
        # Get the root project directory dynamically for subprocess execution
        # This assumes utils.py is in src/, and project root is one level up from src/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Run the transformation script
        subprocess.run(['python3', TRANSFORM_SCRIPT_PATH], check=True, cwd=project_root, capture_output=True)
        st.success("Data transformed!")

        # Run the training script
        subprocess.run(['python3', TRAIN_SCRIPT_PATH], check=True, cwd=project_root, capture_output=True)
        st.success("Model re-trained with new data! Your recommendations are now up-to-date.")

        # Invalidate Streamlit caches that depend on the model/data
        st.cache_data.clear()
        st.cache_resource.clear()

        st.rerun() # Use st.rerun() instead of st.experimental_rerun()

    except subprocess.CalledProcessError as e:
        st.error(f"Error during re-training process: {e.stderr.decode()}")
        st.error("Please check your `src/transform.py` and `src/train_model.py` scripts for errors when run outside the app.")
    except Exception as e:
        st.error(f"An unexpected error occurred during re-training: {e}")