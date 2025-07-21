import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib 
import os
import matplotlib.pyplot as plt

import gspread
import streamlit as st
from utils import get_gsheet_client


TRANSFORMED_GSHEET_URL_KEY = "transformed_google_sheet"
TRANSFORMED_GSHEET_TAB_NAME = "Processed Data"
MODEL_DIR = "models"

def load_transformed_workouts_from_gsheet():
    """
    Loads transformed workout data from the specified Google Sheet tab.
    """
    gc = get_gsheet_client()
    print("Google Sheet client gotten successfully for transformed data.")
    try:
        print(f"Opening sheet URL for transformed data using key: '{TRANSFORMED_GSHEET_URL_KEY}'...")
        sheet_url = st.secrets[TRANSFORMED_GSHEET_URL_KEY]["url"]
        spreadsheet = gc.open_by_url(sheet_url)
        print("Transformed spreadsheet opened from URL.")
        worksheet = spreadsheet.worksheet(TRANSFORMED_GSHEET_TAB_NAME)
        print(f"Worksheet '{TRANSFORMED_GSHEET_TAB_NAME}' found successfully.")
        data = worksheet.get_all_records()
        print("Got all records from transformed GSheet.")
        df = pd.DataFrame(data)

        if not df.empty:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)
            # Ensure numeric columns are actually numeric
            numeric_cols = ['weight_lbs', 'sets', 'reps', 'rpe', 'volume', 'target_reps', 'reps_over_target', 'ready_for_increase', 'next_weight_lbs','exercise_encoded']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['weight_lbs', 'sets', 'reps'], inplace=True)
            df.columns = df.columns.str.strip()
            print(f"Loaded {len(df)} rows from Transformed Google Sheet '{TRANSFORMED_GSHEET_TAB_NAME}'.")
            print(f"Columns found in transformed data: {df.columns.tolist()}")
        else:
            print(f"No data found in Transformed Google Sheet '{TRANSFORMED_GSHEET_TAB_NAME}'. Returning empty DataFrame.")
        return df
    except KeyError as e:
        st.error(f"Transformed Google Sheet URL not found in secrets (key: '{TRANSFORMED_GSHEET_URL_KEY}.url'): {e}. Please check your `secrets.toml` configuration.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading transformed data from Google Sheet: {e}")
        return pd.DataFrame()

def load_exercise_encoder():
    """Loads the LabelEncoder saved by transform.py."""
    encoder_path = os.path.join(MODEL_DIR, 'exercise_label_encoder.joblib')
    if os.path.exists(encoder_path):
        try:
            le = joblib.load(encoder_path)
            print(f"Loaded existing LabelEncoder from {encoder_path}")
            return le
        except Exception as e:
            print(f"Error loading LabelEncoder from {encoder_path}: {e}")
            return None
    else:
        print(f"LabelEncoder not found at {encoder_path}. It should be created by transform.py.")
        return None
    
def preprocess_for_training(df):
    """
    prepares features (X) and target (y).
    """
    df = df.copy()
    columns_to_drop = ['Unnamed: 0', 'notes', 'date']

    if 'exercise_encoded' not in df.columns:
        raise KeyError("'exercise_encoded' column not found in transformed data. "
                       "Please ensure transform.py generates and uploads it.")

    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped '{col}' column for training.")

    # Prepare features (X) and target (y)
    X = df.drop(columns=['next_weight_lbs', 'exercise', 'weight_lbs'])
    y = df["next_weight_lbs"]

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Features used for training: {X.columns.tolist()}")

    return X, y

def train_regressor_model(X, y):
    """
    Trains a RandomForestRegressor model and evaluates its OOB score.
    """
    print("Training RandomForestRegressor model...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
    regressor.fit(X, y)
    print("Model training complete.")

    print(f"Out-of-Bag (OOB) Score: {regressor.oob_score_:.4f}")

    return regressor


def evaluate_model_performance(regressor, X, y):
    """
    Evaluates model performance and prints feature importances and prediction results.
    """
    y_pred = regressor.predict(X)

    results_df = pd.DataFrame({
        "Actual_next_weight": y,
        "Predicted_next_weight": y_pred,
        "Residual": y - y_pred # Difference between actual and predicted
    })
    print("\nSample of Actual vs. Predicted Weights:")
    print(results_df.head())
    print("\nFeature Importances:")
    feature_importances = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_importances)

    plt.figure(figsize=(10, 6))
    feature_importances.head(10).plot(kind='barh') # Top 10 features
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("plots/feature_importances.png")
    plt.close()
    print("Feature importances plot saved to plots/feature_importances.png")


if __name__ == "__main__":
    MODEL_DIR = "models"

    # 1. Load transformed data
    st.set_page_config(layout="wide")
    df = load_transformed_workouts_from_gsheet()

    # 2. Preprocess data for training
    X, y = preprocess_for_training(df)

    # 3. Train the model
    regressor = train_regressor_model(X, y)

    # 4. Evaluate model performance
    evaluate_model_performance(regressor, X, y)

    # 5. Save the trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(regressor, os.path.join(MODEL_DIR, 'trained_regressor_model.joblib'))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'model_features.joblib'))
    print(f"\nModel, encoder, and feature names saved to {MODEL_DIR}")

    print("Model training and evaluation complete.")



    

