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

def recommend_next_weight(last_weight, rpe, reps, exercise, target_reps):
    increment = DEFAULT_PROGRESSIONS.get(exercise, 2.5)
    if reps >= target_reps + 2 and rpe <= 7:
        return last_weight + increment
    elif rpe >= 9 and reps < target_reps:
        return last_weight - increment
    else:
        return last_weight


        

