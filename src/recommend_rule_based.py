"""
recommend_rule_based.py

Purpose:
--------
Contains logic for weight progression and future workout recommendations
in an AI-powered personal training system.

Includes:
---------
- DEFAULT_PROGRESSIONS: Default weight increments per exercise
- recommend_next_weight(): Suggests weight changes based on user performance

Planned:
--------
- recommend_workout(): Will generate beginner routines based on goals, training days, and equipment
"""

import pandas as pd
import numpy as np

DEFAULT_PROGRESSIONS = {
    "Squat": 5.0,
    "Deadlift": 5.0,
    "Bench Press": 2.5,
    "Overhead Press": 1.25,
    "Barbell Row": 2.5,
    "Bicep Curl": 1.0,
    "Lat Pulldown": 2.5
}
"""
LOGIC:
if the user can do 2 more than the target for a specified exercise then up the weight
if too difficult for the user then drop the weight
else keep the current weight
"""

def recommend_next_weight(last_weight, rpe, reps, exercise, target_reps, goal_weight=None):
    """
    Recommends the next workout weight based on a rule-based progressive overload strategy.

    This function applies a '2-for-2' style rule:
    - If the user performed well (e.g., 2+ reps over target with RPE <= 7), weight increases.
    - If the workout was very difficult (RPE >= 9 and reps below target), weight decreases.
    - Otherwise, the weight remains the same.

    Args:
        last_weight (float): The weight lifted in the most recent successful set (in kg).
        rpe (int): Rate of Perceived Exertion for the last set (1-10 scale).
        reps (int): Actual repetitions performed in the last set.
        exercise (str): The name of the exercise (e.g., "Squat", "Bench Press").
        target_reps (int): The target repetitions for the exercise.
        goal_weight (float, optional): An optional upper limit for the recommended weight.
                                       If the recommended weight exceeds this, it will be capped.
                                       Defaults to None.
        progression_threshold_reps (int, optional): How many reps over target
                                                    are needed to consider a weight increase. Defaults to 2.
        rpe_difficulty_threshold (int, optional): The RPE value at or above which
                                                  the workout is considered too difficult. Defaults to 9.

    Returns:
        float: The recommended next weight in kg.
    """
    # defaulting to 2.5 kg if not found.
    increment = DEFAULT_PROGRESSIONS.get(exercise, 2.5)
    recommendation_note = "Maintaining current weight." # Default note

    # If goal is already reached or exceeded, don't increase further
    if goal_weight is not None and last_weight >= goal_weight:
        print(f"Goal weight ({goal_weight}kg) already reached for {exercise}. Maintaining current weight.")
        recommendation_note = f"Goal weight ({goal_weight}kg) already reached. Maintaining current weight."
        return goal_weight, recommendation_note

    # Apply 2-for-2 style logic
    if reps >= target_reps + 2 and rpe <= 7:
        next_weight = last_weight + increment
        recommendation_note = "Excellent! Time for a weight increase (rule-based)."
        # Cap recommendation at goal_weight
        if goal_weight:
            next_weight = min(next_weight, goal_weight)
            if next_weight == goal_weight:
                recommendation_note = f"Excellent! Goal weight ({goal_weight}kg) reached, maintaining."
        return next_weight, recommendation_note

    # 3. Rule for decreasing weight (workout was too difficult)
    elif rpe >= 9 and reps < target_reps:
        next_weight = max(last_weight - increment, 0)  # prevent negative weights
        recommendation_note = f"Workout for {exercise} was difficult. Decreasing weight by {increment}kg (rule-based)."
        return next_weight, recommendation_note
    
    # 4. Default: Maintain current weight
    else:
        return last_weight, recommendation_note

    
# --- Example Usage (for demonstration and testing) ---
if __name__ == "__main__":
    print("--- Rule-Based Weight Recommendations ---")

    # Scenario 1: Ready to increase
    print("\nScenario 1: User crushed Squats (3 reps over target, RPE 6)")
    recommended_squat = recommend_next_weight(
        last_weight=100.0, rpe=6, reps=10, exercise="Squat", target_reps=7
    )
    print(f"Recommended next Squat weight: {recommended_squat:.2f} kg") # Expected: 105.00 kg

    # Scenario 2: Too difficult
    print("\nScenario 2: User struggled with Bench Press (2 reps under target, RPE 9)")
    recommended_bench = recommend_next_weight(
        last_weight=52.5, rpe=6, reps=7, exercise="Bench Press", target_reps=8
    )
    print(f"Recommended next Bench Press weight: {recommended_bench:.2f} kg") # Expected: 77.50 kg

    # Scenario 3: Maintain weight
    print("\nScenario 3: User hit Deadlift target exactly (RPE 8)")
    recommended_deadlift = recommend_next_weight(
        last_weight=150.0, rpe=8, reps=5, exercise="Deadlift", target_reps=5
    )
    print(f"Recommended next Deadlift weight: {recommended_deadlift:.2f} kg") # Expected: 150.00 kg

    # Scenario 4: Goal weight capping
    print("\nScenario 4: User ready to increase but hits goal weight")
    recommended_ohp = recommend_next_weight(
        last_weight=40.0, rpe=6, reps=7, exercise="Overhead Press", target_reps=5, goal_weight=42.0
    )
    print(f"Recommended next Overhead Press weight: {recommended_ohp:.2f} kg") # Expected: 42.00 kg (capped)

    print("\nRecommendation examples complete.")

