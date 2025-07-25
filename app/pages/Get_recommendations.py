import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import random

import gspread
from src.utils import get_gsheet_client
from src.recommend_rule_based import recommend_next_weight, DEFAULT_PROGRESSIONS

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
EXERCISE_INFO_DEFAULTS = {
    'target_reps': (8, 12),
    'min_inc': 2.5,
    'max_inc': 5.0,
    'deload_pct': 0.85,
    'max_rpe_for_inc': 7,
    'min_rpe_to_consider_fail': 8
}
known_exercise_info = {
    'Bench Press': {'target_reps': (8, 12), 'min_inc': 2.5, 'max_inc': 5.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Overhead Press': {'target_reps': (6, 10), 'min_inc': 2.5, 'max_inc': 2.5, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Bicep Curl': {'target_reps': (8, 15), 'min_inc': 2.5, 'max_inc': 2.5, 'deload_pct': 0.80, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Lat Pulldown': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Seated Row': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Hack Squat': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8}
}

# --- Session State Initialization ---
if 'show_log_form' not in st.session_state:
    st.session_state.show_log_form = False
if 'recommended_data' not in st.session_state:
    st.session_state.recommended_data = {}

@st.cache_data(ttl=timedelta(minutes=5))
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
            df.dropna(subset=['weight_lbs', 'sets', 'reps', 'rpe'], inplace=True)
            df.columns = df.columns.str.strip()

        return df
    except KeyError as e:
        st.error(f"Transformed Google Sheet URL not found in secrets (key: '{TRANSFORMED_GSHEET_URL_KEY}.url'): {e}. Please check your `secrets.toml` configuration.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading transformed data from Google Sheet: {e}")
        return pd.DataFrame()

# --- Load trained model and encoder ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
print(f"Attempting to load models from: {os.path.abspath(MODEL_DIR)}")
try:
    regressor = joblib.load(os.path.join(MODEL_DIR, 'trained_regressor_model.joblib'))
    exercise_encoder = joblib.load(os.path.join(MODEL_DIR, 'exercise_label_encoder.joblib'))
    model_features = joblib.load(os.path.join(MODEL_DIR, 'model_features.joblib'))
    df_history = load_transformed_workouts_from_gsheet()
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
    for ex in st.session_state.all_exercise_options:
        if ex not in st.session_state.exercise_info_dynamic:
            st.session_state.exercise_info_dynamic[ex] = EXERCISE_INFO_DEFAULTS.copy()

# --- Function to prepare input for prediction ---
@st.cache_data
def prepare_input_for_ml_prediction(exercise_name, weight_lbs, reps, sets, rpe, _encoder, features_list):
    input_data = pd.DataFrame([{
        'weight_lbs': float(weight_lbs),
        'reps': int(reps),
        'sets': int(sets),
        'rpe': int(rpe),
        'exercise': exercise_name
    }])
    if exercise_name not in _encoder.classes_:
        st.info(f"Adding '{exercise_name}' to the model's known exercises (for this session).")
        existing_classes = list(_encoder.classes_)
        existing_classes.append(exercise_name)
        _encoder.fit(existing_classes)
        input_data['exercise_encoded'] = _encoder.transform([exercise_name])
    else:
        input_data['exercise_encoded'] = _encoder.transform([exercise_name])
    if 'rir' in features_list:
        input_data['rir'] = 10 - input_data['rpe']
    if 'reps_x_rpe' in features_list:
        input_data['reps_x_rpe'] = input_data['reps'] * input_data['rpe']
    input_data = input_data.drop(columns=['exercise'])
    final_input_df = pd.DataFrame(columns=features_list)
    for col in features_list:
        if col in input_data.columns:
            final_input_df[col] = input_data[col]
        else:
            final_input_df[col] = 0.0
    for col in final_input_df.columns:
        final_input_df[col] = pd.to_numeric(final_input_df[col], errors='coerce')
        final_input_df[col] = final_input_df[col].fillna(0)
    return final_input_df

def log_workout(date, exercise, weight_lbs, reps, sets, rpe, notes=""):
    print(f"DEBUG: log_workout called for {exercise} on {date}")
    try:
        gc = get_gsheet_client()
        print("DEBUG: Google Sheets client obtained.")
        raw_sheet_url = st.secrets["google_sheet"]["url"]
        print(f"DEBUG: Raw sheet URL: {raw_sheet_url}")
        spreadsheet = gc.open_by_url(raw_sheet_url)
        print(f"DEBUG: Spreadsheet '{spreadsheet.title}' opened.")
        worksheet = spreadsheet.worksheet(RAW_WORKOUT_SHEET_NAME)
        print(f"DEBUG: Worksheet '{RAW_WORKOUT_SHEET_NAME}' selected.")

        new_row = [
            date.strftime('%Y-%m-%d'),
            exercise,
            float(weight_lbs),
            int(sets),
            int(reps),
            int(rpe),
            notes
        ]
        print(f"DEBUG: New row data prepared: {new_row}")

        worksheet.append_row(new_row)
        print("DEBUG: Row append attempt finished.")

        st.success("Workout logged successfully!")
        st.info("Re-processing data and re-training model in the background. This may take a moment...")
        load_transformed_workouts_from_gsheet.clear()
        st.warning("Note: Data transformation and model re-training typically happen as a separate, scheduled process for efficiency. "
                   "For immediate reflection, you'll need to manually re-run the `transform.py` and `train_model.py` scripts on your server/local machine, or set up a CI/CD pipeline on Streamlit Cloud.")
        st.rerun()

    except KeyError:
        print("DEBUG: KeyError - raw sheet URL not found in secrets.")
        st.error("Google Sheet URL for raw data not found in secrets. Please check your `secrets.toml` configuration.")
    except Exception as e:
        print(f"DEBUG: General error logging workout: {e}")
        st.error(f"Error logging workout: {e}")


# --- Function for post-prediction rules ---
# def apply_post_prediction_rules(predicted_weight_raw, current_weight, current_reps, current_rpe, exercise_name):
#     info = st.session_state.exercise_info_dynamic.get(exercise_name, EXERCISE_INFO_DEFAULTS)
#     recommended_weight = predicted_weight_raw
#     recommendation_note = "ML-Based Recommendation"
#     MAX_ALLOWABLE_DELOAD_PCT = 0.70
#     if current_rpe <= 5 and current_reps >= info['target_reps'][1]:
#         inc_options = [info['max_inc']]
#         if info['max_inc'] * 1.5 <= 100:
#              inc_options.append(round((info['max_inc']*1.5)/info['min_inc'])*info['min_inc'])
#         recommended_weight = current_weight + random.choice(inc_options)
#         recommendation_note = "Excellent! Time for a significant weight increase."
#     elif current_rpe <= 7 and current_reps >= info['target_reps'][0]:
#         if predicted_weight_raw <= current_weight + info['min_inc'] * 0.5:
#              recommended_weight = current_weight + info['min_inc']
#              recommendation_note = "Great effort! A small increase is recommended."
#         else:
#             recommended_weight = predicted_weight_raw
#             recommendation_note = "Great effort! A good increase is recommended."
#     elif current_rpe >= 8:
#         if current_reps >= info['target_reps'][0]:
#             if predicted_weight_raw >= current_weight:
#                 recommended_weight = current_weight + random.choice([0, info['min_inc']])
#                 recommendation_note = "Pushing hard! Maintain or slight increase."
#             else:
#                 recommended_weight = current_weight
#                 recommendation_note = "Solid effort. Maintain weight to build strength."
#         else:
#             forced_deload_weight = current_weight * info['deload_pct']
#             recommended_weight = max(predicted_weight_raw, forced_deload_weight)
#             recommended_weight = max(recommended_weight, current_weight * MAX_ALLOWABLE_DELOAD_PCT)
#             recommendation_note = "Challenging session. Consider a deload to recover."
#     min_weight_floor = current_weight * MAX_ALLOWABLE_DELOAD_PCT
#     recommended_weight = max(recommended_weight, min_weight_floor)
#     recommended_weight = round(recommended_weight / 2.5) * 2.5
#     min_allowed_exercise_weight = 20.0 if exercise_name not in ['Bench Press', 'Hack Squat'] else 45.0
#     recommended_weight = max(recommended_weight, min_allowed_exercise_weight)
#     return recommended_weight, recommendation_note

# --- Consolidated Function for applying post-prediction rules and generating notes ---
import random # Make sure random is imported if not already

def apply_post_prediction_rules(predicted_weight_raw, current_weight, current_reps, current_rpe, exercise_name):
    """
    Applies post-prediction rules to the raw ML prediction, potentially adjusting it,
    and generates a recommendation note.
    """
    # Assuming st.session_state and DEFAULT_PROGRESSIONS are accessible
    # from the context where this function is called (e.g., app/pages/Get_recommendation.py)
    info = st.session_state.exercise_info_dynamic.get(exercise_name, EXERCISE_INFO_DEFAULTS)
    
    # Initialize final_recommended_weight. We'll always set it explicitly within the rules.
    # We still keep predicted_weight_raw for scenarios where the ML is specifically intended to be used.
    final_recommended_weight = float(predicted_weight_raw) # Start with ML, but rule-based will override as needed
    recommendation_note = "ML-Based Recommendation."

    # Get the typical increment/decrement for this exercise
    increment_val = DEFAULT_PROGRESSIONS.get(exercise_name, 2.5)

    # Define MAX_ALLOWABLE_DELOAD_PCT here once or globally if used across files
    MAX_ALLOWABLE_DELOAD_PCT = 0.70
    min_weight_floor_by_percent = current_weight * MAX_ALLOWABLE_DELOAD_PCT

    # --- Core Logic for Blending ML and Rules ---

    # Scenario 1: Very easy set, significant increase
    if current_rpe <= 5 and current_reps >= info['target_reps'][1]:
        inc_options = [info['max_inc']]
        if info['max_inc'] * 1.5 <= 100:
             inc_options.append(round((info['max_inc']*1.5)/info['min_inc'])*info['min_inc'])
        
        rule_based_increase = current_weight + random.choice(inc_options)
        final_recommended_weight = max(predicted_weight_raw, rule_based_increase)
        recommendation_note = "Excellent! Time for a significant weight increase."

    # Scenario 2: Good performance (RPE <= 7), but potentially missed reps
    elif current_rpe <= 7:
        if current_reps >= info['target_reps'][0]:
            # RPE is good, reps hit target or more. Progress.
            if predicted_weight_raw <= current_weight + info['min_inc'] * 0.5:
                # ML is conservative, nudge up by min_inc
                final_recommended_weight = current_weight + info['min_inc']
                recommendation_note = "Great effort! A small increase is recommended."
            else:
                # Trust ML for a good increase
                final_recommended_weight = predicted_weight_raw
                recommendation_note = "Great effort! A good increase is recommended."
        else: # <--- THIS IS THE CRUCIAL 'ELSE' BLOCK FOR SCENARIO 2
            # RPE is good (<=7), but reps are BELOW target.
            # This implies the weight is slightly too heavy, even if RPE isn't maxed.
            final_recommended_weight = current_weight - increment_val # Suggest a small, controlled decrease
            recommendation_note = "Reps slightly low for good RPE. Consider a small weight decrease."


    # Scenario 3: Challenging session (RPE >= 8)
    elif current_rpe >= 8:
        if current_reps >= info['target_reps'][0]:
            # Pushing hard, but hit reps. Maintain or very slight increase.
            final_recommended_weight = current_weight + random.choice([0, info['min_inc']])
            recommendation_note = "Pushing hard! Maintain or slight increase (rule-based)."
        else: # RPE >= 8 AND reps < target_reps (User struggled, didn't hit reps)
            # Non-ML logic for decrease based on reps deficit
            reps_deficit = info['target_reps'][0] - current_reps
            
            if reps_deficit >= 3: # If failed by 3 or more reps, suggest a larger deload
                final_recommended_weight = current_weight * info['deload_pct']
                recommendation_note = "Significant struggle. Deload recommended for recovery (rule-based)."
            else: # For minor struggles (failed by 1 or 2 reps)
                final_recommended_weight = current_weight - increment_val
                recommendation_note = "Challenging session. Consider a small weight decrease to hit reps (rule-based)."

    # Default case if none of the specific increase/decrease rules above were met
    # This acts as a fallback for scenarios not explicitly covered, e.g.,
    # RPE 7, but current_reps is just barely below info['target_reps'][0] and
    # the explicit `else` for Scenario 2 was missed (which it shouldn't be now).
    # Or, if you want ML to take over *only* when rules don't apply.
    # Given your current structure, this block might be hit less now that Scenario 2 is complete.
    else:
        # If no specific rule has overridden the initial predicted_weight_raw,
        # and it's not a clear increase/decrease scenario, then default to current weight.
        # This prevents very low ML predictions from persisting if no other rule caught them.
        final_recommended_weight = current_weight
        recommendation_note = "Maintaining current weight (rule-based, no clear progression/regression)."


    # --- Final Rounding and Safety Checks (Apply to all recommendations) ---
    # These should apply *after* the primary recommendation logic, to ensure practical values.

    # 1. Ensure weight doesn't go below the absolute floor (e.g., 70% of current weight)
    final_recommended_weight = max(final_recommended_weight, min_weight_floor_by_percent)
    
    # 2. Round to nearest valid plate increment (e.g., 2.5 lbs)
    final_recommended_weight = round(final_recommended_weight / 2.5) * 2.5

    # 3. Enforce minimum absolute exercise weight (e.g., empty bar, or specific for exercise)
    min_allowed_exercise_weight = 20.0 if exercise_name not in ['Bench Press', 'Hack Squat'] else 45.0
    final_recommended_weight = max(final_recommended_weight, min_allowed_exercise_weight)

    return final_recommended_weight, recommendation_note


# --- UI Layout ---
st.title("🏋️ Workout AI Coach")
st.markdown("Get your next workout weight recommendation based on your historical performance.")

# Input form for Recommendation
with st.form("recommendation_form"):
    selected_exercise = st.selectbox(
        "**Select or Add Exercise:**",
        options=st.session_state.all_exercise_options,
        key="exercise_select",
        help="Type to filter, or type a new exercise name and press Enter to add it.",
        index=0 if st.session_state.all_exercise_options else None,
        accept_new_options=True
    )

    if selected_exercise not in st.session_state.all_exercise_options and selected_exercise:
        st.session_state.all_exercise_options.append(selected_exercise)
        st.session_state.all_exercise_options.sort()
        if selected_exercise not in st.session_state.exercise_info_dynamic:
            st.session_state.exercise_info_dynamic[selected_exercise] = EXERCISE_INFO_DEFAULTS.copy()
        st.rerun()

    last_workout_for_selected_exercise = None
    if selected_exercise and selected_exercise in df_history['exercise'].unique():
        last_workout_for_selected_exercise = df_history[df_history['exercise'] == selected_exercise].sort_values(by='date', ascending=False).iloc[0]

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
        input_df = prepare_input_for_ml_prediction(
            selected_exercise,
            current_weight,
            current_reps,
            current_sets,
            current_rpe,
            exercise_encoder,
            model_features
        )

        if input_df is not None:
            try:
                predicted_weight_raw = regressor.predict(input_df)[0]
                recommended_weight, recommendation_note = apply_post_prediction_rules(
                    predicted_weight_raw, current_weight, current_reps, current_rpe, selected_exercise
                )

                st.markdown("---")
                st.subheader(f"💪 Recommendation for {selected_exercise}:")
                st.markdown(f"**Next Recommended Weight: <span style='font-size: 36px; color: #28a745;'>{recommended_weight:.2f} lbs</span>**", unsafe_allow_html=True)
                st.info(recommendation_note)

                # Store recommendation data in session state for the next form
                st.session_state.recommended_data = {
                    "exercise": selected_exercise,
                    "recommended_weight": recommended_weight,
                    "current_reps": current_reps,
                    "current_sets": current_sets,
                    "current_rpe": current_rpe
                }
                st.session_state.show_log_form = True
                #st.rerun()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure your model is trained and data is correctly formatted.")
        else:
            st.warning("Please select or add an exercise.")
    else: # If no exercise selected/added for recommendation
        st.warning("Please select or add an exercise to get a recommendation.")


# --- Log This Workout Form---
if st.session_state.show_log_form:
    st.markdown("### Log This Workout (Optional)")
    st.info("You can adjust the suggested values if your actual workout differed.")

    with st.form("log_recommended_workout_form", clear_on_submit=True):
        log_date = st.date_input("Date:", value=datetime.today(), key="log_rec_date")

        # Pre-fill with recommended weight/reps/sets from session_state
        log_weight = st.number_input(
            "Actual Weight (lbs):",
            min_value=0.0, value=st.session_state.recommended_data.get("recommended_weight", 50.0), step=2.5, key="log_rec_weight"
        )
        log_reps = st.number_input(
            "Actual Reps:",
            min_value=1, value=st.session_state.recommended_data.get("current_reps", 8), step=1, key="log_rec_reps"
        )
        log_sets = st.number_input(
            "Actual Sets:",
            min_value=1, value=st.session_state.recommended_data.get("current_sets", 3), step=1, key="log_rec_sets"
        )
        log_rpe = st.slider(
            "Actual RPE (1-10):",
            min_value=1, max_value=10, value=st.session_state.recommended_data.get("current_rpe", 7), step=1, key="log_rec_rpe"
        )
        log_notes = st.text_area("Notes (optional):", key="log_rec_notes")

        log_submitted = st.form_submit_button("Log Recommended Workout")

        if log_submitted:
            print("DEBUG: 'Log Recommended Workout' button clicked. Initiating log_workout.")
            log_workout(
                log_date,
                st.session_state.recommended_data["exercise"],
                log_weight,
                log_reps,
                log_sets,
                log_rpe,
                log_notes
            )
            print("DEBUG: log_workout function finished.")

# --- Historical Context for the selected exercise ---
if 'selected_exercise' in locals() and selected_exercise and selected_exercise in df_history['exercise'].unique():
    st.markdown("---")
    st.subheader(f"📊 Your Recent Progress for {selected_exercise}:")

    exercise_history = df_history[df_history['exercise'] == selected_exercise].tail(10).sort_values(by='date', ascending=True)

    if not exercise_history.empty:
        st.dataframe(exercise_history[['date', 'weight_lbs', 'reps', 'rpe']].set_index('date'))
        st.line_chart(exercise_history.set_index('date')['weight_lbs'])
        st.caption("Weight Progression Over Time")
    else:
        st.info("No historical data available for this exercise yet. Input a few sessions to see your progress!")
elif 'selected_exercise' in locals() and selected_exercise and selected_exercise not in df_history['exercise'].unique():
    st.info(f"No historical data yet for '{selected_exercise}'. After you log some sessions, your progress will appear here!")

st.markdown("---")
st.caption("Developed with Streamlit and Scikit-learn. Data-driven workout insights.")