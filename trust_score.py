import numpy as np
import joblib

def compute_sensor_stability(data_dict):
    """
    Computes a simple stability metric based on deviation
    Higher = more unstable → higher trust score (more confidence in issue)
    """
    deviation = np.std(list(data_dict.values()))
    stability = min(deviation / 10, 1.0)  # normalized 0–1
    return stability


def compute_driving_pattern_score(cluster_id):
    """
    Cluster interpretation (can customize):
    0 = Calm
    1 = Normal
    2 = Aggressive
    """
    mapping = {
        0: 0.2,   # Calm driver → low failure chance
        1: 0.5,   # Balanced driver
        2: 0.9    # Aggressive driver → higher risk
    }
    return mapping.get(cluster_id, 0.5)


def compute_trust_score(failure_prob, driving_cluster, is_anomaly, sensor_dict):
    # Base components
    driving_score = compute_driving_pattern_score(driving_cluster)
    stability_score = compute_sensor_stability(sensor_dict)
    anomaly_score = 1.0 if is_anomaly else 0.0

    # Weighted formula
    trust_score = (
        failure_prob * 0.50 +
        driving_score * 0.20 +
        anomaly_score * 0.20 +
        stability_score * 0.10
    )

    # Normalize 0–100
    trust_score = round(trust_score * 100, 2)
    return trust_score
