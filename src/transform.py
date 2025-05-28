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

def calculate_features(df):
    """
    Calculates derived features like 'volume' and 'next_weight_kg'.
    Assumes 'exercise', 'date', 'weight_kg', 'sets', 'reps' columns exist.
    """
    df = df.copy()

    # Ensure correct volume calculation (weight * sets * reps)
    if 'volume' not in df.columns:
        df['volume'] = df['weight_kg'] * df['sets'] * df['reps']
        print("Calculated 'volume' feature.")

    # Sort and calculate next_weight_kg for training target
    df = df.sort_values(["exercise", "date"]).reset_index(drop=True)
    df["next_weight_kg"] = df.groupby("exercise")["weight_kg"].shift(-1)
    print("Calculated 'next_weight_kg' target column.")

    return df

def clean_data(df):
    """
    Performs data cleaning steps.
    """
    df = df.copy()
    # Drop rows where next_weight_kg is NaN (these are the last entries for each exercise)
    initial_rows = len(df)
    df.dropna(subset=["next_weight_kg"], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing 'next_weight_kg' (last entry for each exercise).")

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
    plt.plot(exercise_df["date"], exercise_df["weight_kg"], marker='o', linestyle='-')
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

if __name__ == "__main__":
    # Define file paths
    RAW_DATA_PATH = "data/raw_workout_log.csv"
    TRANSFORMED_DATA_PATH = "data/transformed_workout_log.csv"
    PLOTS_DIR = "plots" 
    # 1. Load data
    workout_log_df = load_raw_data(RAW_DATA_PATH)

    # 2. Calculate features
    workout_log_df = calculate_features(workout_log_df)

    # 3. Clean data
    workout_log_df = clean_data(workout_log_df)

    # 4. Save transformed data
    os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)
    workout_log_df.to_csv(TRANSFORMED_DATA_PATH, index=False)
    print(f"\nTransformed data saved to: {TRANSFORMED_DATA_PATH}")
    print(f"Final transformed data shape: {workout_log_df.shape}")
    print("Sample of transformed data:")
    print(workout_log_df.head())


    # 5. Visualize trends for key exercises
    unique_exercises = workout_log_df['exercise'].unique()
    for exercise in unique_exercises:
        visualize_trend(workout_log_df, exercise, output_dir=PLOTS_DIR)

    print("\nData transformation and visualization complete.")
