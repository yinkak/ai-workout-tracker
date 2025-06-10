# src/recommend_model_based.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Useful if you want a separate test set
import joblib # For saving/loading models and encoders
import os
import matplotlib.pyplot as plt
from train_model import *

from src.recommend_rule_based import recommend_next_weight as recommend_next_weight_rules_based
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

def preprocess_input_data(
        TRANSFORMED_DATA_PATH, exercise_encoder : LabelEncoder, model_features : list
        )-> pd.DataFrame:    
    """
    Applies label encoding and prepares features (X) and target (y).
    """
    if not os.path.exists(TRANSFORMED_DATA_PATH):
        print(f"Error: CSV file not found at {TRANSFORMED_DATA_PATH}")
    
    else:
    # Load the CSV file into a pandas DataFrame
        input_data = pd.read_csv(TRANSFORMED_DATA_PATH)
        print("CSV loaded successfully!")
        print(input_data.head()) # Print the first few rows to verify

    # Drop columns not needed for training or already processed
    # 'Unnamed: 0' often comes from saving/loading CSVs without index=False
    columns_to_drop = ['Unnamed: 0', 'notes', 'date'] # 'date' is usually not a direct feature for next_weight prediction

    for col in columns_to_drop:
        if col in input_data.columns:
            input_data = input_data.drop(columns=[col])
            print(f"Dropped '{col}' column for training.")

    # Encode categorical features
    # 'exercise' needs to be encoded before dropping the original column
    input_data["exercise_encoded"] = exercise_encoder.fit_transform(df["exercise"])
    print("Encoded 'exercise' column.")

    input_data = input_data[model_features]

def predict_next_weight_ml(
    trained_model : RandomForestRegressor,
    input_df : pd.DataFrame
) -> float:
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




    


    

