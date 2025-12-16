import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import joblib

# --------------------------------------
# LOAD DATA
# --------------------------------------
df = pd.read_csv("vehicle_telematics_demo.csv")

# Features for failure prediction
features = [
    "engine_temp",
    "vibration_level",
    "brake_temp",
    "speed",
    "acceleration",
    "harsh_braking_count",
    "mileage",
    "aggressiveness_score"
]

X = df[features]
y = df["failure_label"]

# --------------------------------------
# SCALE FEATURES
# --------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# --------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------------------
# MODEL 1: FAILURE PREDICTION MODEL
# --------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

# Save the model
joblib.dump(rf, "failure_prediction_model.pkl")

# Evaluate
y_pred = rf.predict(X_test)
print("\n==== FAILURE PREDICTION MODEL REPORT ====")
print(classification_report(y_test, y_pred))

# --------------------------------------
# MODEL 2: ANOMALY DETECTION (Isolation Forest)
# --------------------------------------
iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

iso.fit(X_scaled)

joblib.dump(iso, "anomaly_detection_model.pkl")

df["anomaly_score"] = iso.decision_function(X_scaled)
df["is_anomaly"] = iso.predict(X_scaled)  # -1 = anomaly

# --------------------------------------
# MODEL 3: DRIVING PATTERN CLUSTERING (USP)
# --------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
driving_features = df[["speed", "acceleration", "harsh_braking_count", "aggressiveness_score"]]

kmeans.fit(driving_features)

df["driving_pattern_cluster"] = kmeans.predict(driving_features)

joblib.dump(kmeans, "driving_pattern_model.pkl")

print("\n==== DRIVING PATTERN CLUSTERS ====")
print(df["driving_pattern_cluster"].value_counts())

# --------------------------------------
# SAVE FINAL DATA WITH LABELS
# --------------------------------------
df.to_csv("vehicle_telematics_processed.csv", index=False)

print("\nModels trained and saved successfully:")
print("✔ failure_prediction_model.pkl")
print("✔ anomaly_detection_model.pkl")
print("✔ driving_pattern_model.pkl")
print("✔ scaler.pkl")
print("✔ vehicle_telematics_processed.csv")
