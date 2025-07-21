# src/utils.py

import streamlit as st
import pandas as pd
import os
import subprocess
from datetime import datetime
import gspread

# --- File Paths (Relative to project root, as fixed before) ---
RAW_WORKOUT_LOG_PATH = "data/raw_workout_log.csv"
TRANSFORM_SCRIPT_PATH = "src/transform.py"
TRAIN_SCRIPT_PATH = "src/train_model.py"

RAW_WORKOUT_SHEET_NAME= "50-Day_Workout_Log"

# Cached function to get Google Sheets client
@st.cache_resource(ttl=3600) # Cache for 1 hour to prevent re-auth on every rerun
def get_gsheet_client():
    try:
        # Authenticate with Google Sheets using st.secrets
        return gspread.service_account_from_dict(st.secrets["gcp_service_account"])
    except Exception as e:
        st.error(f"Error authenticating with Google Sheets: {e}")
        st.stop()

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


# Function to append a new row to Google Sheet
def append_row_to_gsheet(data_row_dict):
    gc = get_gsheet_client()
    sheet_url = st.secrets["google_sheet"]["url"]
    try:
        spreadsheet = gc.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet(RAW_WORKOUT_SHEET_NAME)
        # Append row as a list, ensure order matches sheet columns
        # This assumes data_row_dict has keys in correct order or you explicitly order them
        # E.g., worksheet.append_row([data_row_dict['date'], data_row_dict['exercise'], ...])
        # Simpler: If your sheet has headers, get them and map your dict
        headers = worksheet.row_values(1) # Get headers from the first row
        ordered_values = [data_row_dict.get(header, '') for header in headers]
        worksheet.append_row(ordered_values)
        return True
    except Exception as e:
        st.error(f"Error appending workout to Google Sheet: {e}")
        return False
    
def load_raw_workouts_from_gsheet():
    gc = get_gsheet_client()
    print("google sheet client gotten successfully")
    try:
        print("opening sheet url")
        sheet_url = st.secrets["google_sheet"]["url"]
        print("url opened")
        spreadsheet = gc.open_by_url(sheet_url)
        print("spreadsheet opened from url")
        worksheet = spreadsheet.worksheet(RAW_WORKOUT_SHEET_NAME)
        print("worksheet opened from url")
        data = worksheet.get_all_records() # Gets all data as list of dictionaries
        print("got all records from gsheets")
        df = pd.DataFrame(data)
        if not df.empty:
            # Ensure 'date' column is parsed as datetime
            # 'errors='coerce'' will turn unparseable dates into NaT (Not a Time)
            df['date'] = pd.to_datetime(df['date'], errors='coerce') 
            # Drop rows where date parsing failed
            df.dropna(subset=['date'], inplace=True)
            print(f"Loaded {len(df)} rows from Google Sheet '{RAW_WORKOUT_SHEET_NAME}'.")
        else:
            print(f"No data found in Google Sheet '{RAW_WORKOUT_SHEET_NAME}'. Returning empty DataFrame.")
        return df
    except KeyError:
        st.error("Google Sheet URL not found in Streamlit secrets. Please check your `secrets.toml` configuration.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading raw data from Google Sheet: {e}")
        return pd.DataFrame()