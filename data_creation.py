import numpy as np
import pandas as pd
import random

# Number of vehicles and samples per vehicle
NUM_VEHICLES = 10
SAMPLES_PER_VEHICLE = 500

# Seed for reproducibility
np.random.seed(42)

all_data = []

for vid in range(1, NUM_VEHICLES + 1):
    vehicle_id = f"VH_{vid:03d}"
    
    # Base behavior patterns (each vehicle has unique personality)
    base_engine_temp = np.random.normal(85, 5)
    base_vibration = np.random.normal(2.5, 0.3)
    base_brake_temp = np.random.normal(120, 10)
    base_speed = np.random.normal(60, 10)
    base_mileage = np.random.randint(10000, 60000)
    aggressiveness = np.random.uniform(0.2, 0.9)  # USP
    
    for i in range(SAMPLES_PER_VEHICLE):
        
        engine_temp = np.random.normal(base_engine_temp, 4)
        vibration_level = np.random.normal(base_vibration, 0.4)
        brake_temp = np.random.normal(base_brake_temp, 12)
        speed = np.random.normal(base_speed, 15)
        acceleration = np.random.normal(2 * aggressiveness, 0.5)
        harsh_braking = np.random.poisson(3 * aggressiveness)

        # Mileage increases each row
        mileage = base_mileage + i * np.random.randint(1, 5)

        # Inject anomalies (simulate early failure)
        anomaly_factor = 0
        if random.random() < 0.05:
            engine_temp += np.random.uniform(10, 25)
            vibration_level += np.random.uniform(1, 3)
            brake_temp += np.random.uniform(20, 40)
            anomaly_factor = 1
        
        # Failure probability (ground truth)
        failure_prob = (
            0.2 * (engine_temp / 120) +
            0.3 * (vibration_level / 8) +
            0.3 * (brake_temp / 200) +
            0.2 * aggressiveness
        )
        
        failure_prob = min(max(failure_prob, 0), 1)

        failure_label = 1 if (failure_prob > 0.65 or anomaly_factor == 1) else 0

        all_data.append([
            vehicle_id,
            engine_temp,
            vibration_level,
            brake_temp,
            speed,
            acceleration,
            harsh_braking,
            mileage,
            aggressiveness,
            failure_prob,
            failure_label
        ])

# Column names
columns = [
    "vehicle_id",
    "engine_temp",
    "vibration_level",
    "brake_temp",
    "speed",
    "acceleration",
    "harsh_braking_count",
    "mileage",
    "aggressiveness_score",   # USP feature
    "failure_probability_gt",
    "failure_label"
]

df = pd.DataFrame(all_data, columns=columns)

# Save CSV
df.to_csv("vehicle_telematics_demo.csv", index=False)

print("Demo synthetic telematics dataset generated successfully!")
print(df.head())
