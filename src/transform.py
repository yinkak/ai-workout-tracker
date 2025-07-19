# """
# transform.py

# Purpose:
# --------
# Handles data loading, transformation, and visualization for workout logs
# in the AI Personal Trainer system.

# Includes:
# ---------
# - read_sample_csv(): Loads and sorts workout log CSV as a pandas DataFrame
# - visualize_trend(): Plots weight progression over time for a given exercise
# - Volume and personal record (PR) calculations

# Dependencies:
# -------------
# - pandas
# - matplotlib
# - numpy
# """


# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import numpy as np

# #helper function to transform and sort the csv rfile into a dataframe
# def read_sample_csv(workout_log_file):
#     workout_log_df = pd.read_csv(workout_log_file, parse_dates=["date"])
#     workout_log_df = workout_log_df.sort_values('date').reset_index(drop=True)
#     return workout_log_df

# def visualize_trend(exercise, workout_log):
#     df = workout_log[workout_log['exercise'] == exercise] 
#     plt.plot(df["date"], df["weight_kg"])
#     plt.xlim(df["date"].min(), df["date"].max())

#     plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # tick every 1 day
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     plt.xticks(rotation=45)

#     plt.title(f"{exercise} Weight Over Time")
#     plt.show()

# workout_log_df = read_sample_csv("data/sample_workout_log.csv")
# workout_log_df['volume'] = workout_log_df['weight_kg'] * workout_log_df['sets'] * workout_log_df['sets']

# workout_log_df = workout_log_df.sort_values(["exercise", "date"]).reset_index(drop=True)
# workout_log_df["next_weight_kg"] = workout_log_df.groupby("exercise")["weight_kg"].shift(-1)


# print(workout_log_df)
# workout_log_df.to_csv("data/transformed_workout_log.csv")


# #visualize growth
# visualize_trend("Squat", workout_log_df)

# src/transform.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import gspread
import streamlit as st
from utils import get_gsheet_client

RAW_WORKOUT_SHEET_NAME = "50-Day_Workout_Log"
TRANSFORMED_GSHEET_URL_KEY = "transformed_google_sheet" # Key for the new URL in secrets.toml
TRANSFORMED_GSHEET_TAB_NAME = "Processed Data" # The tab name within the new transformed Google Sheet


def load_raw_data(file_path):
    """
    Loads raw workout log CSV into a pandas DataFrame and parses dates.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    print(f"Loaded raw data from: {file_path}")
    print(f"Initial shape: {df.shape}")
    return df

# Function to load all raw data from Google Sheet
def load_raw_workouts_from_gsheet():
    gc = get_gsheet_client()
    # Make sure your Streamlit secrets are correctly configured for 'google_sheet.url'
    # In a deployed Streamlit app, this will come from your app's secrets.toml
    # For local testing, you might need to mock this or have a .streamlit/secrets.toml file
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
        return pd.DataFrame() # Return empty if error

def calculate_features(df):
    """
    Calculates derived features like 'volume' and 'next_weight_lbs'. next_weight_kg
    Assumes 'exercise', 'date', 'weight_lbs', 'sets', 'reps' columns exist.
    """
    df = df.copy()
    #convert weight_kg to weight_lbs


    # Ensure correct volume calculation (weight * sets * reps)
    if 'volume' not in df.columns:
        df['volume'] = df['weight_lbs'] * df['sets'] * df['reps']
        print("Calculated 'volume' feature.")

    #Default mapping for target reps for each exercise
    target_reps_mapping = {
        'Squat': 8,
        'Bench Press': 8,
        'Deadlift': 5, 
        'Overhead Press': 6,
        'Barbell Row': 8,
        'Lat Pulldown': 10,
        'Bicep Curl': 10,
        # Add other exercises from your log with their typical target reps
    }

    if 'target_reps' not in df.columns:
        df['target_reps'] = df['exercise'].apply(lambda x: target_reps_mapping.get(x, 8))
        print("Added 'target_reps' feature based on predefined mapping.")

    # --- Calculate 'reps_over_target' ---
    df['reps_over_target'] = df['reps'] - df['target_reps']
    print("Calculated 'reps_over_target' feature.")

    # --- NEW FEATURE: 'ready_for_increase' ---
    df['ready_for_increase'] = ((df['reps'] >= 12) & (df['rpe'] <= 7)).astype(int)
    print("Calculated 'ready_for_increase' feature.")


    # Sort and calculate next_weight_lbs for training target
    df = df.sort_values(["exercise", "date"]).reset_index(drop=True)
    df["next_weight_lbs"] = df.groupby("exercise")["weight_lbs"].shift(-1)
    print("Calculated 'next_weight_lbs' target column.")

    return df

def clean_data(df):
    """
    Performs data cleaning steps.
    """
    df = df.copy()
    # Drop rows where next_weight_lbs is NaN (these are the last entries for each exercise)
    initial_rows = len(df)
    df.dropna(subset=["next_weight_lbs"], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing 'next_weight_lbs' (last entry for each exercise).")

    # Drop notes if they are not used as a feature
    if 'notes' in df.columns:
        df = df.drop(columns=["notes"])
        print("Dropped 'notes' column.")
    return df

def visualize_trend(df, exercise, output_dir="plots"):
    """
    Plots weight progression over time for a given exercise.
    Saves the plot to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    exercise_df = df[df['exercise'] == exercise].copy() 
    if exercise_df.empty:
        print(f"No data found for exercise: {exercise}. Skipping visualization.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(exercise_df["date"], exercise_df["weight_lbs"], marker='o', linestyle='-')
    plt.xlim(exercise_df["date"].min(), exercise_df["date"].max())

    # Format x-axis for dates
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7)) 
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right') 

    plt.title(f"{exercise} Weight Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Weight (kg)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() 

    plot_filename = os.path.join(output_dir, f"{exercise}_weight_trend.png")
    plt.savefig(plot_filename)
    print(f"Saved plot for {exercise} to {plot_filename}")
    plt.close() 

def upload_dataframe_to_gsheet(df, sheet_url_key, tab_name):
    """
    Uploads a pandas DataFrame to a specified Google Sheet tab.
    It will clear the tab and then write the DataFrame, including headers.
    """
    if df.empty:
        print(f"DataFrame is empty, skipping upload to Google Sheet '{tab_name}'.")
        return

    gc = get_gsheet_client()
    try:
        gsheet_url = st.secrets[sheet_url_key]["url"]
        spreadsheet = gc.open_by_url(gsheet_url)
        print(f"Opened transformed spreadsheet at URL key: {sheet_url_key}")

        try:
            # Try to get the existing worksheet
            worksheet = spreadsheet.worksheet(tab_name)
            print(f"Worksheet '{tab_name}' found. Clearing existing data.")
            worksheet.clear() # Clear all cells in the worksheet
        except gspread.exceptions.WorksheetNotFound:
            # If worksheet doesn't exist, create it
            print(f"Worksheet '{tab_name}' not found. Creating new worksheet.")
            worksheet = spreadsheet.add_worksheet(title=tab_name, rows=df.shape[0]+1, cols=df.shape[1])
            # Note: add_worksheet creates with default rows/cols, can be adjusted later if needed

        # Convert DataFrame to a list of lists, including headers
        data_to_upload = [df.columns.tolist()] + df.values.tolist()

        # Update all cells in the worksheet
        worksheet.update(data_to_upload)
        print(f"Successfully uploaded {len(df)} rows to Google Sheet '{tab_name}'.")

    except KeyError as e:
        st.error(f"Google Sheet URL for transformed data not found in secrets (key: '{sheet_url_key}'): {e}. Please check your `secrets.toml`.")
    except Exception as e:
        st.error(f"Error uploading transformed data to Google Sheet: {e}")

if __name__ == "__main__":
    # Define file paths
    RAW_DATA_PATH = "../data/50-Day_Workout_Log.csv"
    TRANSFORMED_DATA_PATH = "data/transformed_workout_log.csv"
    PLOTS_DIR = "plots" 
    # 1. Load data
    st.set_page_config(layout="wide")
    workout_log_df = load_raw_workouts_from_gsheet()

    # 2. Calculate features
    transformed_df = calculate_features(workout_log_df)

    # 3. Clean data
    transformed_df = clean_data(transformed_df)
    

    # # 4. Save transformed data
    # os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)
    # workout_log_df.to_csv(TRANSFORMED_DATA_PATH, index=False)
    # print(f"\nTransformed data saved to: {TRANSFORMED_DATA_PATH}")
    # print(f"Final transformed data shape: {workout_log_df.shape}")
    # print("Sample of transformed data:")
    # print(workout_log_df.head())

    # Convert datetime columns to string format for Google Sheets upload ---
    print("\nConverting datetime columns to string format for GSheet upload...")
    for col in transformed_df.columns:
        if pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
            transformed_df[col] = transformed_df[col].dt.strftime('%Y-%m-%d')
            # You can choose a different format like '%Y-%m-%d %H:%M:%S' if you need time as well
    print("Datetime columns converted.")

    # 4. Upload transformed data to a new Google Sheet
    print(f"\nUploading transformed data to Google Sheet '{TRANSFORMED_GSHEET_TAB_NAME}'...")
    upload_dataframe_to_gsheet(transformed_df, TRANSFORMED_GSHEET_URL_KEY, TRANSFORMED_GSHEET_TAB_NAME)
    print("Transformed data upload process complete.")


    # 5. Visualize trends for key exercises
    df_for_viz = transformed_df.copy()
    if 'date' in df_for_viz.columns:
            # Ensure the string format matches what you used for strftime
        df_for_viz['date'] = pd.to_datetime(df_for_viz['date'], errors='coerce')


    print("\nGenerating visualizations...")
    if 'exercise' in df_for_viz.columns: # Use df_for_viz here
        unique_exercises = df_for_viz['exercise'].unique()
        for exercise in unique_exercises:
            visualize_trend(df_for_viz, exercise, output_dir=PLOTS_DIR) # Pass df_for_viz
        print("\nData transformation and visualization complete.")
    else:
        print("Cannot visualize trends: 'exercise' column not found in transformed data.")
else:
    print("No raw data loaded from Google Sheet. Skipping transformation and visualization.")


    print("\nData transformation and visualization complete.")
