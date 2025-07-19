import streamlit as st
import pandas as pd
import joblib # To load your model and encoder
import os
from datetime import datetime, timedelta
import random

import gspread
from src.utils import get_gsheet_client

# --- Configuration ---
st.set_page_config(
    page_title="Workout AI Coach",
    page_icon="🏋️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

TRANSFORMED_GSHEET_URL_KEY = "transformed_google_sheet"
TRANSFORMED_GSHEET_TAB_NAME = "Processed Data"
RAW_WORKOUT_SHEET_NAME = "50-Day_Workout_Log"

# Define exercise_info globally or load from a config file
# This will be updated if a new exercise is added by the user
EXERCISE_INFO_DEFAULTS = {
    'target_reps': (8, 12),
    'min_inc': 2.5,
    'max_inc': 5.0,
    'deload_pct': 0.85,
    'max_rpe_for_inc': 7,
    'min_rpe_to_consider_fail': 8
}
# Define specific info for known exercises
# This will be the base, and new exercises will get EXERCISE_INFO_DEFAULTS
known_exercise_info = {
    'Bench Press': {'target_reps': (8, 12), 'min_inc': 2.5, 'max_inc': 5.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Overhead Press': {'target_reps': (6, 10), 'min_inc': 2.5, 'max_inc': 2.5, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Bicep Curl': {'target_reps': (8, 15), 'min_inc': 2.5, 'max_inc': 2.5, 'deload_pct': 0.80, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Lat Pulldown': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Seated Row': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Hack Squat': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8}
}

@st.cache_data(ttl=timedelta(minutes=5)) # Cache for 5 minutes to avoid hitting GSheet too often
def load_transformed_workouts_from_gsheet():
    gc = get_gsheet_client()
    try:
        sheet_url = st.secrets[TRANSFORMED_GSHEET_URL_KEY]["url"]
        spreadsheet = gc.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet(TRANSFORMED_GSHEET_TAB_NAME)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        if not df.empty:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)

            numeric_cols = ['weight_lbs', 'sets', 'reps', 'rpe', 'volume', 'target_reps',
                            'reps_over_target', 'ready_for_increase', 'next_weight_lbs']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['weight_lbs', 'sets', 'reps', 'rpe'], inplace=True) # Critical columns
            df.columns = df.columns.str.strip() # Clean column names

        return df
    except KeyError as e:
        st.error(f"Transformed Google Sheet URL not found in secrets (key: '{TRANSFORMED_GSHEET_URL_KEY}.url'): {e}. Please check your `secrets.toml` configuration.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading transformed data from Google Sheet: {e}")
        return pd.DataFrame()

# --- Load trained model and encoder ---
MODEL_DIR = "models"
print(f"Attempting to load models from: {os.path.abspath(MODEL_DIR)}")
try:
    regressor = joblib.load(os.path.join(MODEL_DIR, 'trained_regressor_model.joblib'))
    exercise_encoder = joblib.load(os.path.join(MODEL_DIR, 'exercise_label_encoder.joblib'))
    model_features = joblib.load(os.path.join(MODEL_DIR, 'model_features.joblib'))
    
    # Load your entire transformed workout log for historical context
    st.warning("loading transformed workouts now")
    df_history = load_transformed_workouts_from_gsheet() # NEW: Load from GSheet
    st.warning("workout history found")
    if df_history.empty:
        st.warning("No historical transformed data loaded from Google Sheet. Predictions might be less accurate or not possible.")


except FileNotFoundError:
    st.error("Model or data files not found! Please ensure 'train_model.py' has been run to train and save the model, and 'transform.py' has processed your data.")
    st.stop()

# Initialize session state for dynamic exercise options and info
if 'all_exercise_options' not in st.session_state:
    st.session_state.all_exercise_options = sorted(df_history['exercise'].unique().tolist())

if 'exercise_info_dynamic' not in st.session_state:
    st.session_state.exercise_info_dynamic = known_exercise_info.copy()
    # Ensure all existing exercises from history are in exercise_info_dynamic
    for ex in st.session_state.all_exercise_options:
        if ex not in st.session_state.exercise_info_dynamic:
            st.session_state.exercise_info_dynamic[ex] = EXERCISE_INFO_DEFAULTS.copy()

# --- Function to prepare input for prediction ---
@st.cache_data # Cache this function for performance
def prepare_input_for_ml_prediction(exercise_name, weight_lbs, reps, sets, rpe, _encoder, features_list): # Changed 'encoder' to '_encoder'
    # Create a DataFrame for the input
    input_data = pd.DataFrame([{
        'weight_lbs': float(weight_lbs), # Ensure float type
        'reps': int(reps), # Ensure int type
        'sets': int(sets), # Ensure int type
        'rpe': int(rpe), # Ensure int type
        'exercise': exercise_name # Use original exercise name for encoding
    }])

    # --- Encoding the exercise ---
    # Temporarily extend the encoder if the new exercise is not known
    if exercise_name not in _encoder.classes_: # Use _encoder here
        st.info(f"Adding '{exercise_name}' to the model's known exercises (for this session).")
        existing_classes = list(_encoder.classes_) # Use _encoder here
        existing_classes.append(exercise_name)
        _encoder.fit(existing_classes) # Refit the encoder with the new class (use _encoder)
        input_data['exercise_encoded'] = _encoder.transform([exercise_name]) # Use _encoder here
    else:
        input_data['exercise_encoded'] = _encoder.transform([exercise_name]) # Use _encoder here

    # Feature Engineering (MUST match train_model.py's preprocess_for_training)
    if 'rir' in features_list:
        input_data['rir'] = 10 - input_data['rpe']
    if 'reps_x_rpe' in features_list:
        input_data['reps_x_rpe'] = input_data['reps'] * input_data['rpe']

    # Drop original 'exercise' column as it's now encoded
    input_data = input_data.drop(columns=['exercise'])

    # Ensure the order of columns matches the training data
    final_input_df = pd.DataFrame(columns=features_list)
    for col in features_list:
        if col in input_data.columns:
            final_input_df[col] = input_data[col]
        else:
            final_input_df[col] = 0.0 # Using float 0.0 for numeric consistency

    for col in final_input_df.columns:
        final_input_df[col] = pd.to_numeric(final_input_df[col], errors='coerce')
        final_input_df[col] = final_input_df[col].fillna(0)

    return final_input_df

def log_workout(date, exercise, weight_lbs, reps, sets, rpe, notes=""):
    print(f"DEBUG: log_workout called for {exercise} on {date}") # Check 1
    try:
        gc = get_gsheet_client()
        print("DEBUG: Google Sheets client obtained.") # Check 2
        raw_sheet_url = st.secrets["google_sheet"]["url"]
        print(f"DEBUG: Raw sheet URL: {raw_sheet_url}") # Check 3
        spreadsheet = gc.open_by_url(raw_sheet_url)
        print(f"DEBUG: Spreadsheet '{spreadsheet.title}' opened.") # Check 4
        worksheet = spreadsheet.worksheet(RAW_WORKOUT_SHEET_NAME)
        print(f"DEBUG: Worksheet '{RAW_WORKOUT_SHEET_NAME}' selected.") # Check 5

        new_row = [
            date.strftime('%Y-%m-%d'),
            exercise,
            float(weight_lbs), 
            int(sets),         
            int(reps),        
            int(rpe),          
            notes
        ]
        print(f"DEBUG: New row data prepared: {new_row}") # Check 6
        
        worksheet.append_row(new_row)
        print("DEBUG: Row append attempt finished.") # Check 7
        st.success("Workout logged successfully!")
        st.info("Re-processing data and re-training model in the background. This may take a moment...")
        load_transformed_workouts_from_gsheet.clear() 
        st.warning("Note: Data transformation and model re-training typically happen as a separate, scheduled process for efficiency. "
                   "For immediate reflection, you'll need to manually re-run the `transform.py` and `train_model.py` scripts on your server/local machine, or set up a CI/CD pipeline on Streamlit Cloud.")
        st.rerun() 
    except KeyError:
        print("DEBUG: KeyError - raw sheet URL not found in secrets.") # Check 8
        st.error("Google Sheet URL for raw data not found in secrets. Please check your `secrets.toml` configuration.")
    except Exception as e:
        print(f"DEBUG: General error logging workout: {e}") # Check 9
        st.error(f"Error logging workout: {e}")


# --- Function for post-prediction rules ---
def apply_post_prediction_rules(predicted_weight_raw, current_weight, current_reps, current_rpe, exercise_name):
    # Get specific info for the current exercise, use defaults if new
    info = st.session_state.exercise_info_dynamic.get(exercise_name, EXERCISE_INFO_DEFAULTS)
    
    recommended_weight = predicted_weight_raw # Initialize with model's raw prediction
    recommendation_note = "ML-Based Recommendation"

    # Define a minimum deload percentage to prevent overly drastic drops
    # Even if model predicts very low, don't drop more than this percentage
    MAX_ALLOWABLE_DELOAD_PCT = 0.70 # e.g., don't go below 70% of current weight
                                    # Adjust based on your preference

    # --- Rule 1: Very easy set, significant increase ---
    if current_rpe <= 5 and current_reps >= info['target_reps'][1]:
        inc_options = [info['max_inc']]
        if info['max_inc'] * 1.5 <= 100:
             inc_options.append(round((info['max_inc']*1.5)/info['min_inc'])*info['min_inc'])
        recommended_weight = current_weight + random.choice(inc_options)
        recommendation_note = "Excellent! Time for a significant weight increase."

    # --- Rule 2: Easy set, small increase (RPE 6-7) ---
    elif current_rpe <= 7 and current_reps >= info['target_reps'][0]:
        # If model predicts lower or same, force a small increase if conditions are good
        if predicted_weight_raw <= current_weight + info['min_inc'] * 0.5: # Allow for slight model prediction below current+min_inc
             recommended_weight = current_weight + info['min_inc']
             recommendation_note = "Great effort! A small increase is recommended."
        # Else, if model predicted a good increase, let it stand
        else:
            recommended_weight = predicted_weight_raw # Trust model if it predicted a sensible increase
            recommendation_note = "Great effort! A good increase is recommended."

    # --- Rule 3: Hard session (RPE >= 8) ---
    elif current_rpe >= 8: # If RPE is high, regardless of reps, we're in a "hard" zone
        # Option A: If reps are good despite high RPE (e.g., hitting target or above)
        if current_reps >= info['target_reps'][0]:
            # If model predicts maintenance or slight increase, keep it
            if predicted_weight_raw >= current_weight:
                recommended_weight = current_weight + random.choice([0, info['min_inc']]) # Maintain or tiny increase
                recommendation_note = "Pushing hard! Maintain or slight increase."
            else: # Model predicts a drop, but reps were good -> force maintenance
                recommended_weight = current_weight
                recommendation_note = "Solid effort. Maintain weight to build strength."
        # Option B: If reps struggled (below target) with high RPE
        else: # current_reps < info['target_reps'][0]
            # Force a deload based on deload_pct
            forced_deload_weight = current_weight * info['deload_pct']
            
            # Take the max of model's prediction and the forced deload
            # This ensures we don't go too low if model predicts something extreme
            recommended_weight = max(predicted_weight_raw, forced_deload_weight)
            recommended_weight = max(recommended_weight, current_weight * MAX_ALLOWABLE_DELOAD_PCT) # Ensure minimum deload cap

            recommendation_note = "Challenging session. Consider a deload to recover."

    # --- Final Safeguard (catch-all for overly drastic model predictions) ---
    # Apply a general floor for the recommended weight based on current weight
    # This prevents the model from recommending absurdly low weights,
    # especially for new exercises or edge cases.
    min_weight_floor = current_weight * MAX_ALLOWABLE_DELOAD_PCT
    recommended_weight = max(recommended_weight, min_weight_floor)


    # Round to nearest 2.5 lbs increment
    recommended_weight = round(recommended_weight / 2.5) * 2.5
    
    # Ensure min allowed weight for the exercise type
    min_allowed_exercise_weight = 20.0 if exercise_name not in ['Bench Press', 'Hack Squat'] else 45.0
    recommended_weight = max(recommended_weight, min_allowed_exercise_weight)

    return recommended_weight, recommendation_note


# --- UI Layout ---
st.title("🏋️ Workout AI Coach")
st.markdown("Get your next workout weight recommendation based on your historical performance.")

# Input form
with st.form("recommendation_form"):
    # Allow adding new options to the selectbox
    selected_exercise = st.selectbox(
        "**Select or Add Exercise:**",
        options=st.session_state.all_exercise_options,
        key="exercise_select",
        help="Type to filter, or type a new exercise name and press Enter to add it.",
        index=0 if st.session_state.all_exercise_options else None, # Default to first or None
        accept_new_options=True
    )

    # Update session state if a new exercise was typed
    if selected_exercise not in st.session_state.all_exercise_options and selected_exercise:
        st.session_state.all_exercise_options.append(selected_exercise)
        st.session_state.all_exercise_options.sort() # Keep sorted
        
        # Add default info for the new exercise
        if selected_exercise not in st.session_state.exercise_info_dynamic:
            st.session_state.exercise_info_dynamic[selected_exercise] = EXERCISE_INFO_DEFAULTS.copy()
        
        # Rerun to update selectbox options and pre-fill logic
        st.rerun()


    # Find the last workout for the selected exercise (could be None if new)
    last_workout_for_selected_exercise = None
    if selected_exercise in df_history['exercise'].unique():
        last_workout_for_selected_exercise = df_history[df_history['exercise'] == selected_exercise].sort_values(by='date', ascending=False).iloc[0]

    # Pre-fill inputs or set defaults for new exercises
    default_weight = float(last_workout_for_selected_exercise['weight_lbs']) if last_workout_for_selected_exercise is not None else 50.0
    default_reps = int(last_workout_for_selected_exercise['reps']) if last_workout_for_selected_exercise is not None else 8
    default_rpe = int(last_workout_for_selected_exercise['rpe']) if last_workout_for_selected_exercise is not None else 7
    default_sets = int(last_workout_for_selected_exercise['sets']) if last_workout_for_selected_exercise is not None else 3

    col1, col2, col3 = st.columns(3)
    with col1:
        current_weight = st.number_input(
            "**Most Recent Weight (lbs):**",
            min_value=0.0,
            value=default_weight,
            step=2.5,
            key="current_weight"
        )
    with col2:
        current_reps = st.number_input(
            "**Reps Performed:**",
            min_value=1,
            value=default_reps,
            step=1,
            key="current_reps"
        )
    with col3:
        current_rpe = st.slider(
            "**Rate of Perceived Exertion (RPE 1-10):**",
            min_value=1,
            max_value=10,
            value=default_rpe,
            step=1,
            key="current_rpe"
        )
    
    current_sets = st.number_input(
        "**Sets Performed (usually 3):**",
        min_value=1,
        value=default_sets,
        step=1,
        key="current_sets"
    )

    submitted = st.form_submit_button("Get My Recommendation")

# --- Recommendation Logic ---
if submitted:
    if selected_exercise:
        # Prepare input data for the model
        input_df = prepare_input_for_ml_prediction(
            selected_exercise,
            current_weight,
            current_reps,
            current_sets,
            current_rpe,
            exercise_encoder, # Pass the (potentially updated) encoder
            model_features
        )

        if input_df is not None: # Only proceed if exercise was recognized
            try:
                # Predict raw weight using the trained model
                predicted_weight_raw = regressor.predict(input_df)[0]

                # Apply post-prediction rules
                recommended_weight, recommendation_note = apply_post_prediction_rules(
                    predicted_weight_raw,
                    current_weight,
                    current_reps,
                    current_rpe,
                    selected_exercise # Pass the selected exercise name
                )
                
                st.markdown("---")
                st.subheader(f"💪 Recommendation for {selected_exercise}:")
                st.markdown(f"**Next Recommended Weight: <span style='font-size: 36px; color: #28a745;'>{recommended_weight:.2f} lbs</span>**", unsafe_allow_html=True)
                st.info(recommendation_note)

                #Option to Log This Workout ---
                st.markdown("### Log This Workout (Optional)")
                st.info("You can adjust the suggested values if your actual workout differed.")
                
                with st.form("log_recommended_workout_form", clear_on_submit=True):
                    log_date = st.date_input("Date:", value=datetime.today(), key="log_rec_date")
                    
                    # Pre-fill with recommended weight/reps/sets, but allow adjustment
                    log_weight = st.number_input(
                        "Actual Weight (lbs):", 
                        min_value=0.0, value=recommended_weight, step=2.5, key="log_rec_weight"
                    )
                    log_reps = st.number_input(
                        "Actual Reps:", 
                        min_value=1, value=current_reps, step=1, key="log_rec_reps"
                    ) # Often you aim for similar reps
                    log_sets = st.number_input(
                        "Actual Sets:", 
                        min_value=1, value=current_sets, step=1, key="log_rec_sets"
                    ) # Often you aim for similar sets
                    log_rpe = st.slider(
                        "Actual RPE (1-10):", 
                        min_value=1, max_value=10, value=current_rpe, step=1, key="log_rec_rpe"
                    ) # RPE is what you actually felt
                    log_notes = st.text_area("Notes (optional):", key="log_rec_notes")

                    log_submitted = st.form_submit_button("Log Recommended Workout")

                    if log_submitted:
                        log_workout(
                            log_date, selected_exercise, log_weight, log_reps, log_sets, log_rpe, log_notes
                        )

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure your model is trained and data is correctly formatted.")
    else:
        st.warning("Please select or add an exercise.")

# --- Historical Context for the selected exercise ---
if selected_exercise and selected_exercise in df_history['exercise'].unique():
    st.markdown("---")
    st.subheader(f"📊 Your Recent Progress for {selected_exercise}:")
    
    exercise_history = df_history[df_history['exercise'] == selected_exercise].tail(10).sort_values(by='date', ascending=True) # Last 10 entries

    if not exercise_history.empty:
        st.dataframe(exercise_history[['date', 'weight_lbs', 'reps', 'rpe']].set_index('date'))

        # Chart for Weight Progression
        st.line_chart(exercise_history.set_index('date')['weight_lbs'])
        st.caption("Weight Progression Over Time")
    else:
        st.info("No historical data available for this exercise yet. Input a few sessions to see your progress!")
elif selected_exercise and selected_exercise not in df_history['exercise'].unique():
    st.info(f"No historical data yet for '{selected_exercise}'. After you log some sessions, your progress will appear here!")


st.markdown("---")
st.caption("Developed with Streamlit and Scikit-learn. Data-driven workout insights.")