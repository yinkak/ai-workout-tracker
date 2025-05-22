import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#helper function to transform and sort the csv rfile into a dataframe
def read_sample_csv(workout_log_file):
    workout_log_df = pd.read_csv(workout_log_file, parse_dates=["date"])
    workout_log_df = workout_log_df.sort_values('date').reset_index(drop=True)
    return workout_log_df

def visualize_trend(exercise, workout_log):
    df = workout_log[workout_log['exercise'] == exercise] 
    plt.plot(df["date"], df["weight_kg"])
    plt.xlim(df["date"].min(), df["date"].max())

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # tick every 1 day
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.title(f"{exercise} Weight Over Time")
    plt.show()

workout_log_df = read_sample_csv("data/sample_workout_log.csv")
workout_log_df['volume'] = workout_log_df['weight_kg'] * workout_log_df['sets'] * workout_log_df['sets']

workout_log_df["is_pr"] = workout_log_df.groupby("exercise")["weight_kg"].transform(
    lambda x: x == x.cummax()
)


print(workout_log_df)


#visualize growth
visualize_trend("Squat", workout_log_df)
