"""
predictor.py

Purpose:
--------
Trains a Random Forest Regressor model to predict the user's next workout weight
based on historical workout data (e.g., weight lifted, reps, RPE, etc.).

Workflow:
---------
1. Loads and cleans the workout log data
2. Generates the target column ('next_weight_kg') by shifting weight values within each exercise
3. Encodes categorical features (e.g., exercise type)
4. Trains a Random Forest Regressor model using features like weight, reps, sets, and RPE
5. Prints the out-of-bag (OOB) score to evaluate model performance

Dependencies:
-------------
- pandas
- matplotlib
- seaborn
- scikit-learn

Note:
-----
The model is currently trained on labeled historical data. In production, it can be used to predict the next recommended weight given a user's most recent set.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("data/sample_workout_log.csv")
df.info()

df = df.sort_values(["exercise", "date"]).reset_index(drop=True)
df["next_weight_kg"] = df.groupby("exercise")["weight_kg"].shift(-1)

df = df.dropna(subset=["next_weight_kg"])
df = df.drop(columns=["date", "notes"])

label_encoder = LabelEncoder()
df["exercise_encoded"] = label_encoder.fit_transform(df["exercise"])
df = df.drop(columns=["exercise"])

X = df.drop(columns=['next_weight_kg'])
y = df['next_weight_kg']


regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)
regressor.fit(X, y)

print(f"OOB Score: {regressor.oob_score_:.2f}")

