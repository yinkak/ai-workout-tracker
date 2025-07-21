import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import gspread
import streamlit as st
import joblib
from utils import get_gsheet_client
from sklearn.preprocessing import LabelEncoder

RAW_WORKOUT_SHEET_NAME = "50-Day_Workout_Log"
TRANSFORMED_GSHEET_URL_KEY = "transformed_google_sheet" 
TRANSFORMED_GSHEET_TAB_NAME = "Processed Data"
PROJECT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "models")

def load_raw_workouts_from_gsheet():
    gc = get_gsheet_client()
    try:
        sheet_url = st.secrets["google_sheet"]["url"]
        spreadsheet = gc.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet(RAW_WORKOUT_SHEET_NAME)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], errors='coerce') 
            df.dropna(subset=['date'], inplace=True)
        else:
            print(f"No data found in Google Sheet '{RAW_WORKOUT_SHEET_NAME}'. Returning empty DataFrame.")
        return df
    except KeyError:
        st.error("Google Sheet URL not found in Streamlit secrets. Please check your `secrets.toml` configuration.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading raw data from Google Sheet: {e}")
        return pd.DataFrame()

def calculate_features(df):
    """
    Calculates derived features like 'volume' and 'next_weight_lbs'. next_weight_kg
    Assumes 'exercise', 'date', 'weight_lbs', 'sets', 'reps' columns exist.
    """
    df = df.copy()
    # --- 1. Label Encoding for Exercise ---
    if 'exercise' in df.columns and not df['exercise'].empty:
        encoder_path = os.path.join(MODELS_DIR, 'exercise_label_encoder.joblib')
        le = None
        
        if os.path.exists(encoder_path):
            try:
                le = joblib.load(encoder_path)
                print(f"Loaded existing LabelEncoder from {encoder_path} for updating.")
            except Exception as e:
                print(f"Error loading existing LabelEncoder: {e}. A new one will be fitted.")
                le = None 

        if le is None:
            le = LabelEncoder()
            print("No existing LabelEncoder found or failed to load. Fitting a new one.")
            df['exercise_encoded'] = le.fit_transform(df['exercise'])
        else:
            all_classes = np.array(list(set(le.classes_).union(df['exercise'].unique())))
            le.fit(all_classes)
            try:
                df['exercise_encoded'] = le.transform(df['exercise'])
                print("Used and potentially updated existing LabelEncoder for 'exercise_encoded'.")
            except ValueError as e:
                print(f"ValueError during transform: {e}. Handling new categories manually.")
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                df['exercise_encoded'] = df['exercise'].map(mapping).fillna(-1).astype(int) # -1 for unseen during transform
        
        joblib.dump(le, encoder_path)
        print(f"Saved/Updated LabelEncoder at {encoder_path}.")
    else:
        df['exercise_encoded'] = -1
        print("Warning: 'exercise' column not found or is empty. 'exercise_encoded' set to -1.")

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
    }

    if 'target_reps' not in df.columns:
        df['target_reps'] = df['exercise'].apply(lambda x: target_reps_mapping.get(x, 8))

    # --- Calculate 'reps_over_target' ---
    df['reps_over_target'] = df['reps'] - df['target_reps']

    # --- Calculate: 'ready_for_increase' ---
    df['ready_for_increase'] = ((df['reps'] >= 12) & (df['rpe'] <= 7)).astype(int)

    df = df.sort_values(["exercise", "date"]).reset_index(drop=True)
    df["next_weight_lbs"] = df.groupby("exercise")["weight_lbs"].shift(-1)

    return df

def clean_data(df):
    """
    Performs data cleaning steps.
    """
    df = df.copy()
    initial_rows = len(df)
    df.dropna(subset=["next_weight_lbs"], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing 'next_weight_lbs' (last entry for each exercise).")

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
            worksheet = spreadsheet.worksheet(tab_name)
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            print(f"Worksheet '{tab_name}' not found. Creating new worksheet.")
            worksheet = spreadsheet.add_worksheet(title=tab_name, rows=df.shape[0]+1, cols=df.shape[1])

        data_to_upload = [df.columns.tolist()] + df.values.tolist()
        worksheet.update(data_to_upload)
        print(f"Successfully uploaded {len(df)} rows to Google Sheet '{tab_name}'.")

    except KeyError as e:
        st.error(f"Google Sheet URL for transformed data not found in secrets (key: '{sheet_url_key}'): {e}. Please check your `secrets.toml`.")
    except Exception as e:
        st.error(f"Error uploading transformed data to Google Sheet: {e}")

if __name__ == "__main__":
    PLOTS_DIR = "plots" 
    # 1. Load data
    st.set_page_config(layout="wide")
    workout_log_df = load_raw_workouts_from_gsheet()

    # 2. Calculate features
    transformed_df = calculate_features(workout_log_df)

    # 3. Clean data
    transformed_df = clean_data(transformed_df)
    
    print("\nConverting datetime columns to string format for GSheet upload...")
    for col in transformed_df.columns:
        if pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
            transformed_df[col] = transformed_df[col].dt.strftime('%Y-%m-%d')
    print("Datetime columns converted.")

    # 4. Upload transformed data to a new Google Sheet
    print(f"\nUploading transformed data to Google Sheet '{TRANSFORMED_GSHEET_TAB_NAME}'...")
    upload_dataframe_to_gsheet(transformed_df, TRANSFORMED_GSHEET_URL_KEY, TRANSFORMED_GSHEET_TAB_NAME)
    print("Transformed data upload process complete.")


    # 5. Visualize trends for key exercises
    df_for_viz = transformed_df.copy()
    if 'date' in df_for_viz.columns:
        df_for_viz['date'] = pd.to_datetime(df_for_viz['date'], errors='coerce')


    print("\nGenerating visualizations...")
    if 'exercise' in df_for_viz.columns:
        unique_exercises = df_for_viz['exercise'].unique()
        for exercise in unique_exercises:
            visualize_trend(df_for_viz, exercise, output_dir=PLOTS_DIR)
        print("\nData transformation and visualization complete.")
    else:
        print("Cannot visualize trends: 'exercise' column not found in transformed data.")
else:
    print("No raw data loaded from Google Sheet. Skipping transformation and visualization.")


    print("\nData transformation and visualization complete.")
