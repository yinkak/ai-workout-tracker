# pages/04_Workout_History.py

import streamlit as st
import pandas as pd
import os

# --- File Paths ---
TRANSFORMED_WORKOUT_LOG_PATH = "../data/transformed_workout_log.csv"
RAW_WORKOUT_LOG_PATH = "../data/raw_workout_log.csv"

def show_workout_history_page():
    st.title("📚 Your Workout History")
    st.markdown("Dive deep into your past workout performance.")

    # Try to load the transformed data first, as it's cleaner
    if os.path.exists(TRANSFORMED_WORKOUT_LOG_PATH):
        try:
            df_history = pd.read_csv(TRANSFORMED_WORKOUT_LOG_PATH)
            df_history['date'] = pd.to_datetime(df_history['date'])
            
            st.subheader("Filter and Explore:")
            
            # Filters
            all_exercises = sorted(df_history['exercise'].unique().tolist())
            selected_exercise_filter = st.selectbox("Filter by Exercise:", ["All Exercises"] + all_exercises)

            col_date_start, col_date_end = st.columns(2)
            with col_date_start:
                min_date = df_history['date'].min()
                max_date = df_history['date'].max()
                start_date = st.date_input("Start Date:", value=min_date, min_value=min_date, max_value=max_date)
            with col_date_end:
                end_date = st.date_input("End Date:", value=max_date, min_value=min_date, max_value=max_date)

            filtered_df = df_history.copy()
            if selected_exercise_filter != "All Exercises":
                filtered_df = filtered_df[filtered_df['exercise'] == selected_exercise_filter]
            
            # Ensure dates are compatible for filtering
            filtered_df = filtered_df[(filtered_df['date'].dt.date >= start_date) & (filtered_df['date'].dt.date <= end_date)]

            if not filtered_df.empty:
                st.subheader(f"Displaying {len(filtered_df)} Historical Entries:")
                # Display relevant columns
                # If 'notes' is in the dataframe, display it. Otherwise, omit.
                display_cols = ['date', 'exercise', 'weight_lbs', 'reps', 'sets', 'rpe']
                if 'notes' in filtered_df.columns:
                    display_cols.append('notes')
                    
                st.dataframe(filtered_df[display_cols].sort_values(by='date', ascending=False).set_index('date'))

                st.subheader("Visualizations:")
                # Optional: Add charts here for overall trends or filtered trends
                st.line_chart(filtered_df.set_index('date')['weight_lbs'])
                st.caption("Weight Progression")
                
                # Example: reps vs RPE
                fig_reps_rpe = plt.figure(figsize=(10, 5))
                sns.scatterplot(data=filtered_df, x='reps', y='rpe', hue='exercise', style='sets')
                plt.title('Reps vs RPE by Exercise')
                st.pyplot(fig_reps_rpe)
                plt.close(fig_reps_rpe) # Close figure

            else:
                st.info("No data matches your filters. Adjust your selections or log more workouts!")

        except pd.errors.EmptyDataError:
            st.warning("Your transformed workout log is empty. Please log some workouts first!")
        except Exception as e:
            st.error(f"An error occurred while loading workout history: {e}")
            st.warning("Ensure your `data/transformed_workout_log.csv` is correctly formatted.")
    else:
        st.warning(f"No transformed workout data found at `{TRANSFORMED_WORKOUT_LOG_PATH}`. Please ensure you have logged workouts and run `src/transform.py`.")
        if os.path.exists(RAW_WORKOUT_LOG_PATH):
            st.info("Raw data found. Consider running `python3 src/transform.py` from your terminal to generate the transformed data.")

# Main entry point for this page (if run directly, for testing)
if __name__ == '__main__':
    import matplotlib.pyplot as plt # Import for local testing
    import seaborn as sns # Import for local testing
    show_workout_history_page()