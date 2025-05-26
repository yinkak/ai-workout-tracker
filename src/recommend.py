"""
recommend.py

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

#if the user can do 2 more than the target for a specified exercise then up the weight
#if too difficult for the user then drop the weight
#else keep the current weight

def recommend_next_weight(last_weight, rpe, reps, exercise, target_reps, goal_weight=None):
    increment = DEFAULT_PROGRESSIONS.get(exercise, 2.5)

    # If goal is already reached or exceeded, don't increase further
    if goal_weight is not None and last_weight >= goal_weight:
        return goal_weight

    # Apply 2-for-2 style logic
    if reps >= target_reps + 2 and rpe <= 7:
        next_weight = last_weight + increment
        # Cap recommendation at goal_weight
        if goal_weight:
            return min(next_weight, goal_weight)
        return next_weight

    elif rpe >= 9 and reps < target_reps:
        return max(last_weight - increment, 0)  # prevent negative weights

    return last_weight

    



#def recommend_workout(goal, training days, equipment)

