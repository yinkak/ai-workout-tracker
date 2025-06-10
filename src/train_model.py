# """
# train_model.py

# Purpose:
# --------
# Trains a Random Forest Regressor model to predict the user's next workout weight
# based on historical workout data (e.g., weight lifted, reps, RPE, etc.).

# Workflow:
# ---------
# 1. Loads and cleans the workout log data
# 2. Generates the target column ('next_weight_kg') by shifting weight values within each exercise
# 3. Encodes categorical features (e.g., exercise type)
# 4. Trains a Random Forest Regressor model using features like weight, reps, sets, and RPE
# 5. Prints the out-of-bag (OOB) score to evaluate model performance

# Dependencies:
# -------------
# - pandas
# - matplotlib
# - seaborn
# - scikit-learn

# Note:
# -----
# The model is currently trained on labeled historical data. In production, it can be used to predict the next recommended weight given a user's most recent set.
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn
# import os
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder


# df = pd.read_csv("data/transformed_workout_log.csv")
# df.info()
# print(df)

# # df = df.sort_values(["exercise", "date"]).reset_index(drop=True)
# # df["next_weight_kg"] = df.groupby("exercise")["weight_kg"].shift(-1)

# df = df.dropna(subset=["next_weight_kg"])
# df = df.drop(columns=["notes"])

# label_encoder = LabelEncoder()
# df["exercise_encoded"] = label_encoder.fit_transform(df["exercise"])
# df["date_encoded"] = label_encoder.fit_transform(df["date"])
# df = df.drop(columns=["exercise", "date"])

# X = df.drop(columns=['next_weight_kg'])
# y = df["next_weight_kg"]


# regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)
# regressor.fit(X, y)

# y_pred = regressor.predict(X)

# results = pd.DataFrame({
#     "Actual": y,
#     "Predicted": y_pred,
#     "residual": y - y_pred
# })

# print(results)


# print(f"OOB Score: {regressor.oob_score_:.2f}")

# importances = regressor.feature_importances_
# feature_names = X.columns  # Automatically grabs column names from your X
# importance_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
# print("Feature Importances:")
# print(importance_df)


# #should be able to use this to recommend the users next workout based on their progression

# src/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Useful if you want a separate test set
import joblib # For saving/loading models and encoders
import os
import matplotlib.pyplot as plt

def load_transformed_data(file_path):
    """
    Loads transformed workout log data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transformed data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded transformed data from: {file_path}")
    print(f"Initial shape: {df.shape}")
    return df

def preprocess_for_training(df):
    """
    Applies label encoding and prepares features (X) and target (y).
    """
    df = df.copy()

    # Drop columns not needed for training or already processed
    # 'Unnamed: 0' often comes from saving/loading CSVs without index=False
    columns_to_drop = ['Unnamed: 0', 'notes', 'date'] # 'date' is usually not a direct feature for next_weight prediction

    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped '{col}' column for training.")

    # Encode categorical features
    # 'exercise' needs to be encoded before dropping the original column
    global exercise_label_encoder # Declare as global to access it later if needed, or pass it around
    exercise_label_encoder = LabelEncoder()
    df["exercise_encoded"] = exercise_label_encoder.fit_transform(df["exercise"])
    print("Encoded 'exercise' column.")

    # Prepare features (X) and target (y)
    X = df.drop(columns=['next_weight_kg', 'exercise']) # Drop original 'exercise' after encoding
    y = df["next_weight_kg"]

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Features used for training: {X.columns.tolist()}")

    return X, y, exercise_label_encoder

def train_regressor_model(X, y):
    """
    Trains a RandomForestRegressor model and evaluates its OOB score.
    """
    print("Training RandomForestRegressor model...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
    # n_jobs=-1 uses all available processors, speeds up training
    regressor.fit(X, y)
    print("Model training complete.")

    print(f"Out-of-Bag (OOB) Score: {regressor.oob_score_:.4f}") # More precision for score

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

    # You could add more evaluation metrics here if you split your data (e.g., RMSE, MAE)
    # from sklearn.metrics import mean_squared_error, mean_absolute_error
    # print(f"Mean Squared Error (MSE): {mean_squared_error(y, y_pred):.2f}")
    # print(f"Mean Absolute Error (MAE): {mean_absolute_error(y, y_pred):.2f}")


    print("\nFeature Importances:")
    feature_importances = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_importances)

    # Optional: Plot feature importances
    plt.figure(figsize=(10, 6))
    feature_importances.head(10).plot(kind='barh') # Top 10 features
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("plots/feature_importances.png") # Save to plots directory
    plt.close()
    print("Feature importances plot saved to plots/feature_importances.png")


if __name__ == "__main__":
    TRANSFORMED_DATA_PATH = "data/transformed_workout_log.csv"
    MODEL_DIR = "models"

    # 1. Load transformed data
    df = load_transformed_data(TRANSFORMED_DATA_PATH)

    # 2. Preprocess data for training
    X, y, exercise_encoder = preprocess_for_training(df)

    # 3. Train the model
    regressor = train_regressor_model(X, y)

    # 4. Evaluate model performance
    evaluate_model_performance(regressor, X, y) # Note: This is on training data, for OOB score is more reliable

    # 5. Save the trained model and encoder
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(regressor, os.path.join(MODEL_DIR, 'trained_regressor_model.joblib'))
    joblib.dump(exercise_encoder, os.path.join(MODEL_DIR, 'exercise_label_encoder.joblib'))
    # Also save the feature names for robust prediction
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'model_features.joblib'))
    print(f"\nModel, encoder, and feature names saved to {MODEL_DIR}")

    print("Model training and evaluation complete.")



    

