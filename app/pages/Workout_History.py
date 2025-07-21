# ai-workout-tracker/pages/04_Workout_History.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# import plotly.express as px


from src.utils import load_raw_workouts_from_gsheet

st.set_page_config(layout="wide", page_title="Workout History")

def show_workout_history_page():
    st.title("📚 Your Workout History")
    st.markdown("Dive deep into your past workout performance.")

    # --- Load Raw Data from Google Sheets ---
    @st.cache_data(ttl=600) # Cache the raw DataFrame for 10 minutes
    def get_raw_workout_data():
        return load_raw_workouts_from_gsheet()

    df_raw = get_raw_workout_data()

    if df_raw.empty:
        st.warning("No workout data found in Google Sheets. Please log some workouts first on the 'Log Workout' page!")
        return # Exit the function if no data

    try:
        df_history = df_raw.copy()

        # Ensure 'date' column is datetime
        if 'date' in df_history.columns:
            df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
            df_history.dropna(subset=['date'], inplace=True)

        required_cols_for_volume = ['weight_lbs', 'reps', 'sets']
        if all(col in df_history.columns for col in required_cols_for_volume):
            for col in required_cols_for_volume:
                df_history[col] = pd.to_numeric(df_history[col], errors='coerce')
            
            df_history.dropna(subset=required_cols_for_volume, inplace=True)
            
            if not df_history.empty:
                df_history['volume'] = df_history['weight_lbs'] * df_history['reps'] * df_history['sets']
            else:
                st.info("No complete data for volume calculation after cleaning.")
        else:
            st.info("Missing columns to calculate 'volume' (need weight_lbs, reps, and sets).")

        if 'rpe' in df_history.columns:
            df_history['rpe'] = pd.to_numeric(df_history['rpe'], errors='coerce')
            df_history.dropna(subset=['rpe'], inplace=True)


        st.subheader("Filter and Explore:")
        
        # Filters
        if 'exercise' in df_history.columns and not df_history['exercise'].empty:
            all_exercises = sorted(df_history['exercise'].unique().tolist())
            selected_exercise_filter = st.selectbox("Filter by Exercise:", ["All Exercises"] + all_exercises)
        else:
            st.info("No exercise data available for filtering.")
            all_exercises = []
            selected_exercise_filter = "All Exercises"

        col_date_start, col_date_end = st.columns(2)
        with col_date_start:
            min_date = df_history['date'].min() if not df_history.empty else datetime.today().date()
            max_date = df_history['date'].max() if not df_history.empty else datetime.today().date()
            start_date = st.date_input("Start Date:", value=min_date, min_value=min_date, max_value=max_date)
        with col_date_end:
            end_date = st.date_input("End Date:", value=max_date, min_value=min_date, max_value=max_date)

        filtered_df = df_history.copy()
        if selected_exercise_filter != "All Exercises":
            filtered_df = filtered_df[filtered_df['exercise'] == selected_exercise_filter]
        
        filtered_df = filtered_df[(filtered_df['date'].dt.date >= start_date) & (filtered_df['date'].dt.date <= end_date)]

        if not filtered_df.empty:
            st.subheader(f"Displaying {len(filtered_df)} Historical Entries:")
            display_cols = ['date', 'exercise', 'weight_lbs', 'reps', 'sets', 'rpe']
            if 'volume' in filtered_df.columns:
                display_cols.append('volume')
            if 'notes' in filtered_df.columns:
                display_cols.append('notes')
                
            display_cols = [col for col in display_cols if col in filtered_df.columns]

            st.dataframe(filtered_df[display_cols].sort_values(by='date', ascending=False).set_index('date'), use_container_width=True)

            st.subheader("Visualizations:")

            # Line chart for Weight Progression
            if 'weight_lbs' in filtered_df.columns and not filtered_df['weight_lbs'].isnull().all():
                df_sorted = filtered_df.sort_values(by='date')
                sns.set_style("darkgrid")
                sns.set_context("notebook") 
                fig, ax = plt.subplots(figsize=(12, 7))
                unique_exercises = df_sorted['exercise'].unique()
                colors = sns.color_palette("bright", n_colors=len(unique_exercises)) 
                exercise_color_map = {exercise: colors[i] for i, exercise in enumerate(unique_exercises)}

                for exercise in unique_exercises:
                    exercise_df = df_sorted[df_sorted['exercise'] == exercise]
                    ax.plot(
                        exercise_df['date'],
                        exercise_df['weight_lbs'],
                        marker='o',
                        linestyle='-',
                        label=exercise,
                        color=exercise_color_map.get(exercise)
                    )
                ax.set_title("Weight Progression Over Time", fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Date", fontsize=12, labelpad=15)
                ax.set_ylabel("Weight (lbs)", fontsize=12, labelpad=15)
                ax.grid(True, linestyle='-', alpha=0.6)
                ax.tick_params(axis='both', which='major', labelsize=10) 

                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45, ha='right')

                if len(unique_exercises) > 0:
                    ax.legend(title='Exercise', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                sns.despine(left=True, bottom=True)
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No weight data for progression chart or all weights are missing in the filtered data.")

        else:
            st.info("No data matches your filters. Adjust your selections or log more workouts!")

    except Exception as e:
        st.error(f"An error occurred while displaying workout history: {e}")
        st.warning("Please ensure your Google Sheet data is correctly formatted. Common issues include non-numeric values in numeric columns like 'weight_lbs', 'reps', 'sets', or 'rpe'.")

# Main entry point for this page
if __name__ == '__main__':
    from datetime import datetime
    show_workout_history_page()