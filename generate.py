import pandas as pd
from datetime import datetime, timedelta
import random
from io import StringIO

# Your existing data as a string (assuming it's loaded from raw_workout_log.csv)
existing_data_str = """date,exercise,weight_lbs,reps,sets,rpe,notes
2025-05-01,Bench Press,95,8,3,6,Form improving
2025-05-01,Overhead Press,25,8,3,8,Bit difficult
2025-05-03,Bicep Curl,20,10,3,8,Solid session
2025-05-03,Lat Pulldown,100,8,3,8,Solid session
2025-05-05,Seated Row,90,8,3,8,Bit difficult
2025-05-05,Hack Squat,200,10,3,8,Could go heavier
2025-05-07,Bench Press,95,10,3,7,Form improving
2025-05-07,Bicep Curl,20,10,3,6,Progressing slowly
2025-05-08,Bench Press,105,6,3,8,Progressing slowly
2025-05-08,Overhead Press,25,10,3,8,Form improving
2025-05-10,Bicep Curl,25,6,3,8,Progressing slowly
2025-05-10,Lat Pulldown,100,10,3,7,Fatigue showing
2025-05-12,Seated Row,90,9,3,6,Solid session
2025-05-12,Hack Squat,210,10,3,8,could go heavier
2025-05-14,Bench Press,105,7,3,8,Solid session
2025-05-14,Bicep Curl,25,8,3,8,Maintaining weight
2025-05-15,Bench Press,105,8,3,7,Progressing slowly
2025-05-15,Overhead Press,25,12,3,9,could go heavier
2025-05-17,Bicep Curl,25,9,3,7,Progressing slowly
2025-05-17,Lat Pulldown,100,12,3,6,Could go heavier
2025-05-19,Seated Row,90,12,3,7,Maintaining weight
2025-05-19,Hack Squat,220,6,3,7,Heavy
2025-05-21,Bench Press,105,10,3,6,Fatigue showing
2025-05-21,Bicep Curl,25,10,3,8,Solid session
2025-05-22,Bench Press,105,11,3,7,could go heavier
2025-05-22,Overhead Press,30,6,3,7,Progressing slowly
2025-05-24,Bicep Curl,25,12,3,6,Form improving
2025-05-24,Lat Pulldown,110,8,3,7,Solid session
2025-05-26,Seated Row,105,7,3,7,Fatigue showing
2025-05-26,Hack Squat,220,8,3,8,Could go heavier
2025-05-28,Bench Press,115,6,3,8,Could go heavier
2025-05-28,Bicep Curl,30,6,3,7,Form improving
2025-05-29,Bench Press,115,8,3,8,Solid session
2025-05-29,Overhead Press,30,8,3,8,Progressing slowly
2025-05-31,Bicep Curl,30,8,3,7,Solid session
2025-05-31,Lat Pulldown,110,10,3,6,Fatigue showing
2025-06-02,Seated Row,105,10,3,8,Progressing slowly
2025-06-02,Hack Squat,220,10,3,7,Form improving
2025-06-04,Bench Press,115,10,3,6,Solid session
2025-06-04,Bicep Curl,30,9,3,8,Could go heavier
2025-06-05,Bench Press,115,10,3,5,Maintaining weight
2025-06-05,Overhead Press,30,10,3,7, Form improving
2025-06-07,Bicep Curl,30,11,3,7,Form improving
2025-06-07,Lat Pulldown,120,8,3,9,Progressing slowly
2025-06-09,Seated Row,105,12,3,5,Progressing slowly
2025-06-09,Hack Squat,220,12,3,6,Progressing slowly
2025-06-11,Bench Press,115,13,3,5,Could go heavier
2025-06-11,Bicep Curl,35,6,3,7,Progressing slowly
2025-06-12,Bench Press,125,6,3,5,Solid Progressing
2025-06-12,Overhead Press,30,12,3,6,Maintaining weight
2025-06-14,Bicep Curl,35.0,8,3,7,Solid session
2025-06-14,Lat Pulldown,120,10,3,8,Progressing slowly
2025-06-16,Seated Row,120,6,3,8,Progressing slowly
2025-06-16,Hack Squat,230,7,3,8,Could go heavier
2025-06-18,Bench Press,125,11,3,6,Hard session
2025-06-18,Bicep Curl,35.0,8,3,6, Progressing slowly
2025-06-20,Hack Squat,230,10,3,7,Progressing slowly
2025-06-18,Bench Press,125,8,3,6,Hard session
"""

# Load existing data to get the last date and current weights
df_existing = pd.read_csv(StringIO(existing_data_str))
df_existing['date'] = pd.to_datetime(df_existing['date'])
df_existing = df_existing.sort_values(by='date').reset_index(drop=True)

last_date = df_existing['date'].max()
print(f"Last recorded date in existing data: {last_date.strftime('%Y-%m-%d')}")

# Initialize current state for each exercise using the last entry from existing data
exercise_states = {}
for exercise_name in df_existing['exercise'].unique():
    last_entry = df_existing[df_existing['exercise'] == exercise_name].sort_values('date', ascending=False).iloc[0]
    exercise_states[exercise_name] = {
        'current_weight': last_entry['weight_lbs'],
        'current_reps': last_entry['reps'],
        'current_rpe': last_entry['rpe'],
        'failed_progress_count': 0, # How many sessions since last good progression
        'deload_triggered': False,
        'deload_sessions': 0 # How many sessions into the deload
    }

# Define more detailed progression rules and typical rep ranges for each exercise
# Also includes realistic increments and deload percentages
exercise_info = {
    'Bench Press': {'target_reps': (8, 12), 'min_inc': 2.5, 'max_inc': 5.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Overhead Press': {'target_reps': (6, 10), 'min_inc': 2.5, 'max_inc': 2.5, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Bicep Curl': {'target_reps': (8, 15), 'min_inc': 2.5, 'max_inc': 2.5, 'deload_pct': 0.80, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Lat Pulldown': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Seated Row': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8},
    'Hack Squat': {'target_reps': (8, 12), 'min_inc': 5.0, 'max_inc': 10.0, 'deload_pct': 0.85, 'max_rpe_for_inc': 7, 'min_rpe_to_consider_fail': 8}
}

# Add any exercises you *want* to generate data for but might not be in your initial small sample.
# For example, if you want to generate data for "Squat" or "Deadlift"
# If they exist in your full raw_workout_log.csv, they will be initialized above.
# If not, add a sensible starting point here:
# if 'Deadlift' not in exercise_states:
#      exercise_states['Deadlift'] = {'current_weight': 225.0, 'current_reps': 5, 'current_rpe': 7, 'failed_progress_count': 0, 'deload_triggered': False, 'deload_sessions': 0}
#      exercise_info['Deadlift'] = {'target_reps': (4, 6), 'min_inc': 10.0, 'max_inc': 15.0, 'deload_pct': 0.80, 'max_rpe_for_inc': 8, 'min_rpe_to_consider_fail': 9}


# Define workout templates to follow your existing pattern
workout_templates = [
    ['Bench Press', 'Overhead Press'],
    ['Bicep Curl', 'Lat Pulldown'],
    ['Seated Row', 'Hack Squat'],
    # Introduce some variations for longer periods
    ['Bench Press', 'Bicep Curl', 'Hack Squat'],
    ['Overhead Press', 'Lat Pulldown', 'Seated Row'],
    ['Bench Press', 'Lat Pulldown', 'Seated Row'], # New combination
    ['Hack Squat', 'Overhead Press', 'Bicep Curl'] # New combination
]

generated_data = []
current_date = last_date + timedelta(days=1)
end_date = current_date + timedelta(days=90) # Generate for roughly 3 months

DELOAD_THRESHOLD = 3 # Number of consecutive "failed progress" sessions before a deload
DELOAD_DURATION_SESSIONS = 1 # How many sessions the deload lasts (per exercise)

while current_date <= end_date:
    # Simulate a workout day
    if random.random() < 0.75: # Higher chance of workout day to generate more data points
        template = random.choice(workout_templates)
        random.shuffle(template) # Shuffle exercises within the template

        for exercise in template:
            if exercise not in exercise_info or exercise not in exercise_states:
                # This should ideally not happen if exercise_states is populated correctly
                continue 

            state = exercise_states[exercise]
            info = exercise_info[exercise]
            
            # Unpack current state
            current_weight = state['current_weight']
            current_reps_val = state['current_reps']
            current_rpe_val = state['current_rpe']
            
            next_weight = current_weight
            next_reps = random.randint(info['target_reps'][0], info['target_reps'][1])
            next_rpe = current_rpe_val # Baseline for calculation
            notes = ""

            # --- Deload Logic ---
            if state['deload_triggered']:
                if state['deload_sessions'] < DELOAD_DURATION_SESSIONS:
                    next_weight = round((current_weight * info['deload_pct']) / info['min_inc']) * info['min_inc']
                    next_reps = random.randint(info['target_reps'][0] + 2, info['target_reps'][1] + 3) # More reps than usual
                    next_rpe = random.randint(5, 6) # Very easy
                    notes = "Deloading session"
                    state['deload_sessions'] += 1
                else: # Deload completed, reset to normal progression, maybe a slight increase or original weight
                    next_weight = round((state['initial_deload_weight'] + random.choice([0, info['min_inc']])) / info['min_inc']) * info['min_inc']
                    next_reps = random.randint(info['target_reps'][0], info['target_reps'][1])
                    next_rpe = random.randint(6, 8)
                    notes = "Post-deload ramp up"
                    state['deload_triggered'] = False
                    state['deload_sessions'] = 0
                    state['failed_progress_count'] = 0 # Reset failure count
            # --- End Deload Logic ---
            else: # Normal progression logic
                if current_rpe_val <= info['max_rpe_for_inc']: # Was easy enough for potential increase (RPE 5-7)
                    if current_reps_val >= info['target_reps'][1] - 1: # Hit high end of rep range
                        # Increase weight
                        inc = random.choice([info['min_inc'], info['max_inc']])
                        next_weight = current_weight + inc
                        next_reps = random.randint(max(info['target_reps'][0], current_reps_val - 2), current_reps_val) # Reps might drop slightly
                        next_rpe = random.randint(min(10, current_rpe_val + 1), min(10, current_rpe_val + 2)) # RPE typically increases
                        notes = "Good progression"
                        state['failed_progress_count'] = 0
                    else: # Reps were not at high end, try to hit more reps at same weight
                        next_reps = min(info['target_reps'][1], current_reps_val + random.randint(1, 2))
                        next_rpe = random.randint(max(5, current_rpe_val - 1), current_rpe_val)
                        notes = "Building volume at current weight"
                        state['failed_progress_count'] = 0 # Consider this progress
                elif current_rpe_val >= info['min_rpe_to_consider_fail']: # Was hard (RPE 8-9)
                    if current_reps_val < info['target_reps'][0] + random.randint(0,1): # Reps significantly dropped or barely hit low end
                        next_weight = current_weight # Maintain or slight decrease
                        next_reps = random.randint(max(info['target_reps'][0] - 1, 4), info['target_reps'][0])
                        next_rpe = random.randint(current_rpe_val, min(10, current_rpe_val + 1)) # RPE stays high or goes higher
                        notes = "Struggling, maintaining weight"
                        state['failed_progress_count'] += 1
                    else: # Hard, but hit reps, so maintain or very small increase
                        next_weight = current_weight + random.choice([0, info['min_inc']])
                        next_reps = random.randint(max(info['target_reps'][0], current_reps_val -1), current_reps_val)
                        next_rpe = random.randint(current_rpe_val, min(10, current_rpe_val + 1))
                        notes = "Pushing hard"
                        state['failed_progress_count'] += 0.5 # Not a full failure, but not easy progression either
                else: # Default for RPE 7 or unhandled cases: maintain
                    next_weight = current_weight
                    next_reps = random.randint(max(info['target_reps'][0], current_reps_val -1), min(info['target_reps'][1], current_reps_val+1))
                    next_rpe = random.randint(max(5, current_rpe_val-1), min(10, current_rpe_val+1))
                    notes = "Consistent session"
                    state['failed_progress_count'] = 0 # Consistency is progress

                # Check if a deload is triggered
                if state['failed_progress_count'] >= DELOAD_THRESHOLD:
                    state['deload_triggered'] = True
                    state['deload_sessions'] = 0
                    state['initial_deload_weight'] = current_weight # Store weight before deload
                    notes = "Triggering deload" # This note will be overwritten in the next iteration for deload session
            
            # --- Final Adjustments & Rounding ---
            # Ensure weight doesn't go below reasonable minimum (e.g., 20 lbs, 45 lbs for bench)
            min_allowed_weight = 20.0 if exercise != 'Bench Press' else 45.0
            next_weight = max(next_weight, min_allowed_weight)
            
            # Round to nearest 2.5 lbs increment
            next_weight = round(next_weight / 2.5) * 2.5
            
            # Cap RPE at 10 and ensure it's at least 5
            next_rpe = min(10, max(5, next_rpe))

            generated_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'exercise': exercise,
                'weight_lbs': next_weight,
                'reps': next_reps,
                'sets': 3, # Assuming 3 sets always
                'rpe': next_rpe,
                'notes': notes
            })

            # Update current state for the next workout of this exercise
            state['current_weight'] = next_weight
            state['current_reps'] = next_reps
            state['current_rpe'] = next_rpe
            
    current_date += timedelta(days=random.choice([1, 2])) # Move to next day or skip a day

# Convert to DataFrame
new_df = pd.DataFrame(generated_data)

# Print the new data in CSV format, without header (to append)
print("\n--- Artificially Generated Workout Data (append this to your raw_workout_log.csv) ---")
print(new_df.to_csv(index=False, header=False))

# Optional: Print the full combined data (for review, don't append this header if adding to file)
# full_df = pd.concat([df_existing, new_df], ignore_index=True)
# print("\n--- Full Combined Data (Existing + Generated) ---")
# print(full_df.to_csv(index=False))