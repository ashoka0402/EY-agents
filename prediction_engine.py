# prediction_engine.py
import joblib
import numpy as np
import pandas as pd
from trust_score import compute_trust_score  # use your existing trust_score.py

# ---------------------------
# CONFIG / FEATURE ORDERS
# ---------------------------
FEATURE_ORDER = [
    "engine_temp",
    "vibration_level",
    "brake_temp",
    "speed",
    "acceleration",
    "harsh_braking_count",
    "mileage",
    "aggressiveness_score"
]

DRIVING_FEATURES = ["speed", "acceleration", "harsh_braking_count", "aggressiveness_score"]

# ---------------------------
# LOAD MODELS & ARTIFACTS
# ---------------------------
scaler = joblib.load("scaler.pkl")
rf = joblib.load("failure_prediction_model.pkl")
iso = joblib.load("anomaly_detection_model.pkl")
kmeans = joblib.load("driving_pattern_model.pkl")

# Load processed historical dataset for explanation baseline
try:
    hist_df = pd.read_csv("vehicle_telematics_processed.csv")
except Exception:
    hist_df = None  # explanation generator will fall back to simple thresholds if missing

# ---------------------------
# HELPERS
# ---------------------------
def prepare_feature_df(data_dict):
    """
    Returns a pandas DataFrame with a single row and columns matching FEATURE_ORDER.
    This avoids sklearn warnings about feature names.
    """
    row = [data_dict.get(col) for col in FEATURE_ORDER]
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def prepare_driving_df(data_dict):
    """Return DataFrame with driving features only (KMeans expects 4 features)."""
    row = [data_dict.get(col) for col in DRIVING_FEATURES]
    return pd.DataFrame([row], columns=DRIVING_FEATURES)


# ---------------------------
# MAIN PREDICTION FUNCTION
# ---------------------------
def run_full_prediction(data_dict):
    """
    Input example:
    {
        "engine_temp": 95,
        "vibration_level": 3.1,
        "brake_temp": 150,
        "speed": 72,
        "acceleration": 2.8,
        "harsh_braking_count": 4,
        "mileage": 45000,
        "aggressiveness_score": 0.7
    }
    """
    # ensure feature ordering and pass DataFrame to scaler to keep feature names
    feat_df = prepare_feature_df(data_dict)
    # scaler was trained on a DataFrame -> transform with a DataFrame to avoid warnings
    scaled = scaler.transform(feat_df)

    # FAILURE PREDICTION
    failure_pred = int(rf.predict(scaled)[0])
    failure_prob = float(rf.predict_proba(scaled)[0][1])

    # ANOMALY (IsolationForest) : iso was trained on scaled features
    anomaly_flag = int(iso.predict(scaled)[0] == -1)

    # DRIVING PATTERN CLUSTER
    driving_df = prepare_driving_df(data_dict)
    driving_cluster = int(kmeans.predict(driving_df)[0])

    # TRUST SCORE (uses sensor dict for stability)
    trust_score = compute_trust_score(
        failure_prob=failure_prob,
        driving_cluster=driving_cluster,
        is_anomaly=anomaly_flag,
        sensor_dict={k: data_dict[k] for k in FEATURE_ORDER}  # use same keys
    )

    result = {
        "failure_prediction": failure_pred,
        "failure_probability": round(failure_prob, 3),
        "is_anomaly": anomaly_flag,
        "driving_pattern_cluster": driving_cluster,
        "trust_score": trust_score
    }
    return result


# ---------------------------
# EXPLANATION GENERATOR
# ---------------------------
def generate_explanation(result, data_dict, top_n=3):
    """
    Produces a short human-readable explanation for the prediction.
    Uses:
      - anomaly flag
      - z-scores vs history (if history available)
      - simple threshold checks as fallback
    Returns: list of explanation bullet strings.
    """
    bullets = []

    # 1) anomaly message
    if result.get("is_anomaly", 0):
        bullets.append("Anomaly detected in recent sensor readings (unusual spike).")

    # 2) driving pattern interpretation
    cluster_map = {
        0: "calm driving",
        1: "normal driving",
        2: "aggressive driving"
    }
    dp_text = cluster_map.get(result.get("driving_pattern_cluster"), "typical driving")
    bullets.append(f"Driving pattern: {dp_text} (affects wear-and-tear).")

    # 3) feature-based contributors: use z-score vs historical means if available
    contributors = []
    if hist_df is not None:
        # compute baseline mean/std from hist_df
        for feat in ["engine_temp", "vibration_level", "brake_temp", "harsh_braking_count"]:
            if feat in hist_df.columns:
                mu = hist_df[feat].mean()
                sigma = hist_df[feat].std() if hist_df[feat].std() > 0 else 1.0
                val = data_dict.get(feat)
                z = (val - mu) / sigma
                # consider strong contributor if z > 1.5
                if z > 1.5:
                    contributors.append((feat, val, round(z, 2)))
    else:
        # fallback thresholds
        if data_dict.get("engine_temp", 0) > 100:
            contributors.append(("engine_temp", data_dict.get("engine_temp"), None))
        if data_dict.get("brake_temp", 0) > 160:
            contributors.append(("brake_temp", data_dict.get("brake_temp"), None))
        if data_dict.get("vibration_level", 0) > 4.0:
            contributors.append(("vibration_level", data_dict.get("vibration_level"), None))

    # Format top contributors
    if contributors:
        for feat, val, z in contributors[:top_n]:
            if z is not None:
                bullets.append(f"* {feat.replace('_',' ').title()} is high (value={val}, z={z}).")
            else:
                bullets.append(f"* {feat.replace('_',' ').title()} is high (value={val}).")

    # 4) model probability summary
    bullets.append(f"Model failure probability: {result.get('failure_probability')*100:.1f}% (Trust Score: {result.get('trust_score')}%)")

    # 5) recommended action (simple rules)
    if result.get("trust_score") >= 80:
        bullets.append("Recommended action: Immediate service booking (High priority).")
    elif result.get("trust_score") >= 50:
        bullets.append("Recommended action: Schedule service soon (Medium priority).")
    else:
        bullets.append("Recommended action: Monitor and re-evaluate on next drive (Low priority).")

    return bullets


# ---------------------------
# DEMO / CLI
# ---------------------------
if __name__ == "__main__":
    # sample input
    sample = {
        "engine_temp": 102,
        "vibration_level": 3.4,
        "brake_temp": 180,
        "speed": 85,
        "acceleration": 3.2,
        "harsh_braking_count": 6,
        "mileage": 47000,
        "aggressiveness_score": 0.78
    }

    res = run_full_prediction(sample)
    print("PREDICTION RESULT:", res)
    expl = generate_explanation(res, sample)
    print("\nEXPLANATION:")
    for line in expl:
        print("-", line)
