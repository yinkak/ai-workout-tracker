import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os

# Import gspread and your utility to get the client
import gspread
from src.utils import get_gsheet_client # Make sure src/utils.py contains get_gsheet_client

# --- Google Sheet Configuration ---
RAW_WORKOUT_SHEET_NAME = "50-Day_Workout_Log"
TRANSFORMED_GSHEET_URL_KEY = "transformed_google_sheet" # From your secrets.toml
TRANSFORMED_GSHEET_TAB_NAME = "Processed Data"

# --- Function to log workout to Google Sheet ---
def log_workout(date, exercise, weight_lbs, reps, sets, rpe, notes):
    try:
        gc = get_gsheet_client()
        raw_sheet_url = st.secrets["google_sheet"]["url"]
        spreadsheet = gc.open_by_url(raw_sheet_url)
        worksheet = spreadsheet.worksheet(RAW_WORKOUT_SHEET_NAME)

        new_row = [
            date.strftime('%Y-%m-%d'),
            exercise,
            float(weight_lbs),
            int(sets),
            int(reps),
            int(rpe),
            notes
        ]
        worksheet.append_row(new_row)

        st.success("Workout logged successfully to Google Sheet!")
        st.markdown("---")
        st.write("Newly added entry:")
        display_df = pd.DataFrame([new_row], columns=['date', 'exercise', 'weight_lbs', 'sets', 'reps', 'rpe', 'notes'])
        st.dataframe(display_df)

        st.info("Your workout has been logged. For updated recommendations and history, the AI Coach needs to re-process and re-train with this new data.")
        st.warning("Note: Data transformation and model re-training typically happen as a separate, scheduled process (e.g., via GitHub Actions, a cron job on your server, or manually triggering scripts). The deployed Streamlit app *cannot* directly run these processes in a way that immediately updates its internal model files.")
        st.cache_data.clear()
        st.cache_resource.clear()

        st.rerun()

    except gspread.exceptions.APIError as e:
        st.error(f"Google Sheets API Error: {e.response.json().get('error', {}).get('message', 'Unknown API error')}")
        st.error("Please check if your Google Service Account has 'Editor' permissions to the Google Sheet.")
    except KeyError as e:
        st.error(f"Configuration Error: Missing secret key in Streamlit Cloud. Please ensure '{e}' is correctly set in your app's secrets.")
    except Exception as e:
        st.error(f"An unexpected error occurred while logging the workout: {e}")

# --- Function to load existing exercises from Google Sheet for the selectbox ---
@st.cache_data(ttl=timedelta(minutes=5))
def load_existing_exercises_from_gsheet():
    try:
        gc = get_gsheet_client()
        raw_sheet_url = st.secrets["google_sheet"]["url"]
        spreadsheet = gc.open_by_url(raw_sheet_url)
        worksheet = spreadsheet.worksheet(RAW_WORKOUT_SHEET_NAME)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        if not df.empty and 'exercise' in df.columns:
            return sorted(df['exercise'].unique().tolist())
        return []
    except KeyError as e:
        st.warning(f"Could not load existing exercises from Google Sheet: Missing secret '{e}'. Please ensure 'google_sheet_url' is configured.")
        return []
    except Exception as e:
        st.warning(f"Could not load existing exercises from Google Sheet: {e}. Defaulting to empty list.")
        return []


# --- Log Workout Page UI ---
def show_log_workout_page():
    st.title("➕ Log Your Workout")
    st.markdown("Enter the details of your just-completed workout to update your progress.")
    existing_exercises = load_existing_exercises_from_gsheet()
    default_exercise = st.session_state.get('recommended_data', {}).get('exercise', existing_exercises[0] if existing_exercises else None)
    default_weight = st.session_state.get('recommended_data', {}).get('recommended_weight', 50.0)
    default_reps = st.session_state.get('recommended_data', {}).get('current_reps', 8)
    default_sets = st.session_state.get('recommended_data', {}).get('current_sets', 3)
    default_rpe = st.session_state.get('recommended_data', {}).get('current_rpe', 7)


    with st.form("log_workout_form", clear_on_submit=True):
        log_date = st.date_input("Workout Date", datetime.now())

        logged_exercise = st.selectbox(
            "**Exercise:**",
            options=existing_exercises,
            index=existing_exercises.index(default_exercise) if default_exercise in existing_exercises else (0 if existing_exercises else None),
            accept_new_options=True,
            key="logged_exercise_select"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            logged_weight = st.number_input(
                "**Weight (lbs):**",
                min_value=0.0,
                value=default_weight,
                step=2.5,
                key="logged_weight"
            )
        with col2:
            logged_reps = st.number_input(
                "**Reps:**",
                min_value=1,
                value=default_reps,
                step=1,
                key="logged_reps"
            )
        with col3:
            logged_rpe = st.slider(
                "**RPE (1-10):**",
                min_value=1,
                max_value=10,
                value=default_rpe,
                step=1,
                key="logged_rpe"
            )
        
        logged_sets = st.number_input(
            "**Sets:**",
            min_value=1,
            value=default_sets,
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

if __name__ == '__main__':
    show_log_workout_page()