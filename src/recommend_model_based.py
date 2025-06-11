# src/recommend_model_based.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Useful if you want a separate test set
import joblib # For saving/loading models and encoders
import os
import matplotlib.pyplot as plt
import joblib

MIN_WORKOUTS_FOR_ML = 5 
MODEL_DIR= "models"
TRANSFORMED_DATA_PATH = "data/transformed_workout_log.csv"


def load_ml_prediction_assets(model_dir: str):
    """
    Loads the trained ML model, label encoder, and feature names from the specified directory.

    Args:
        model_dir (str): The directory where model assets are stored (e.g., "models").

    Returns:
        tuple: (RandomForestRegressor, LabelEncoder, list of str) or (None, None, None) if loading fails.
    """
    try:
       
        regressor = joblib.load(os.path.join(model_dir, 'trained_regressor_model.joblib'))
        print("Loaded trained_regressor_model.joblib")
        exercise_label_encoder = joblib.load(os.path.join(model_dir, 'exercise_label_encoder.joblib'))
        print("Loaded exercise_label_encoder.joblib")
        model_features = joblib.load(os.path.join(model_dir, 'model_features.joblib'))
        print("Loaded model_features.joblib")

        print(f"Successfully loaded ML model assets from {model_dir}")
        return regressor, exercise_label_encoder, model_features

    except FileNotFoundError as e:
        print(f"Error loading ML prediction assets: {e}. "
              "Make sure you have run train_model.py first and that files exist in '{model_dir}'.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading ML assets: {e}")
        return None, None, None

def get_most_recent_workout_data(exercise_name: str, data_path: str):
    """
    Retrieves the most recent workout data for a specific exercise from the transformed log.
    Also returns the count of historical workouts for that exercise.
    """
    try:
        df = pd.read_csv(data_path, parse_dates=['date']).sort_values(by='date', ascending=False)
        exercise_df = df[df['exercise'].str.lower() == exercise_name.lower()]

        if exercise_df.empty:
            return None, 0

        # Return the most recent workout row and the total count of workouts for this exercise
        return exercise_df.iloc[0], exercise_df.shape[0]
    except FileNotFoundError:
        print(f"Error: Transformed data file not found at {data_path}.")
        return None, 0
    except Exception as e:
        print(f"Error getting most recent workout data: {e}")
        return None, 0
    

def prepare_input_for_ml_prediction(
        current_workout_series: pd.Series, exercise_encoder : LabelEncoder, model_features : list
        )-> pd.DataFrame:    
    """"
    Prepares a single row of workout data for ML model prediction, ensuring
    correct feature encoding and order.

    Args:
        current_workout_features_series (pd.Series): Series with the current workout data
                                                 (e.g., 'weight_kg', 'reps', 'sets', 'rpe', 'exercise').
        exercise_encoder (LabelEncoder): The LabelEncoder fitted on 'exercise' names.
        model_features (list): List of feature names in the order the model expects.

    Returns:
        pd.DataFrame: A DataFrame with the prepared input row, or None if preparation fails.
    """
    if current_workout_series is None or current_workout_series.empty:
        print("Error: No current workout features provided for ML input preparation.")
        return None
    
    #change series into a 2D dataframe for model processing
    # Crucial: .T to transpose from (features as rows) to (features as columns)
    input_data_df =  current_workout_series.to_frame().T
    
    # Calculate 'volume' if it's not present (should be from transform.py)
    if 'volume' not in input_data_df.columns:
        if all(col in input_data_df.columns for col in ['weight_kg', 'sets', 'reps']):
             input_data_df['volume'] = input_data_df['weight_kg'] * input_data_df['sets'] * input_data_df['reps']
        else:
             print("Warning: Cannot calculate 'volume'. Missing 'sets', 'reps', or 'weight_kg'.")
             return None

    # Encode the exercise using the *loaded* encoder
    try:
        # Pass the single exercise value as a list-like to transform
        input_data_df['exercise_encoded'] = exercise_encoder.transform([input_data_df['exercise'].iloc[0]])
    except ValueError:
        print(f"Error: Exercise '{input_data_df['exercise'].iloc[0]}' not recognized by the ML model encoder. "
              "Cannot prepare input for ML model.")
        return None

    # Drop columns not needed for training or already processed
    # 'Unnamed: 0' often comes from saving/loading CSVs without index=False
    columns_to_drop = ['Unnamed: 0', 'notes', 'date'] # 'date' is usually not a direct feature for next_weight prediction

    for col in columns_to_drop:
        if col in input_data_df.columns:
            input_data_df = input_data_df.drop(columns=[col])
            print(f"Dropped '{col}' column for training.")

    # Reindex to ensure feature order matches training data
    try:
        #check for features present in the model that are not present in the input data
        missing_features = set(model_features) - set(input_data_df.columns)
        if missing_features:
            print(f"Error: Missing features for ML model: {missing_features}. Cannot predict.")
            return None
        input_data_df = input_data_df[model_features]
    except KeyError as e:
        print(f"Error: Feature mismatch or ordering issue for ML model. Details: {e}. "
              "Ensure input data contains all features the model was trained on and in correct order.")
        return None

    return input_data_df

def recommend_next_weight_ml_based(
    trained_model : RandomForestRegressor,
    input_df : pd.DataFrame
) -> float :
    """
    Predicts the next weight using the trained ML model.

    Args:
        trained_model (RandomForestRegressor): The loaded ML model.
        input_df (pd.DataFrame): A DataFrame containing one row of pre-processed
                                          and correctly structured input features for the model.

    Returns:
        float: The predicted next weight in kg, or None if prediction fails.
    """
     
    if input_df is None or input_df.empty:
        print("Error: No prepared input data for ML prediction.")
        return None
    
    try:
        predicted_weight = trained_model.predict(input_df)[0]
        return predicted_weight
    except Exception as e:
        print(f"An error occurred during ML prediction: {e}")
        return None


# --- Main execution block for testing this script directly ---
if __name__ == "__main__":
    print("===== Running ML-Based Recommendation Script =====")

    # Make sure you have run src/train_model.py at least once
    # to generate the necessary model assets in the 'models' directory.

    # 1. Load ML assets once at the start of the script's execution
    ml_regressor, ml_exercise_encoder, ml_model_features = load_ml_prediction_assets(MODEL_DIR)

    if ml_regressor is None or ml_exercise_encoder is None or ml_model_features is None:
        print("\nCould not load ML assets. Please ensure 'train_model.py' has been run successfully.")
        sys.exit(1) # Exit if essential assets can't be loaded

    # --- Test Scenarios ---
    print("\n--- Test Scenarios for ML Recommendations ---")

    test_exercises = ["Squat", "Bench Press", "Deadlift", "Bicep Curl", "New Exercise (Insufficient Data)"]

    for exercise in test_exercises:
        print(f"\n--- Attempting ML Recommendation for '{exercise}' ---")

        recent_workout_series, num_workouts = get_most_recent_workout_data(exercise, TRANSFORMED_DATA_PATH)

        if recent_workout_series is None:
            print(f"No historical data found for '{exercise}'. Cannot provide ML recommendation.")
            continue # Skip to the next exercise

        if num_workouts < MIN_WORKOUTS_FOR_ML:
            print(f"Insufficient data ({num_workouts} workouts) for '{exercise}'. "
                  f"ML requires at least {MIN_WORKOUTS_FOR_ML}. Skipping ML recommendation.")
            continue # Skip to the next exercise

        print(f"Most recent workout data for '{exercise}':")
        print(f"  Weight: {recent_workout_series.get('weight_kg')}kg, "
              f"Reps: {recent_workout_series.get('reps')}, "
              f"RPE: {recent_workout_series.get('rpe')}")

        prepared_input_df = prepare_input_for_ml_prediction(
            recent_workout_series, ml_exercise_encoder, ml_model_features
        )

        if prepared_input_df is None:
            print(f"Failed to prepare input for '{exercise}'. ML recommendation aborted.")
            continue

        recommended_weight = recommend_next_weight_ml_based(ml_regressor, prepared_input_df)

        if recommended_weight is not None:
            print(f"ML-Based Recommended next weight for '{exercise}': **{recommended_weight:.2f} kg**")
        else:
            print(f"ML-Based recommendation failed for '{exercise}'.")

    print("\n===== ML-Based Recommendation Script Finished =====")
    


    

