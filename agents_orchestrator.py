"""
Agentic orchestration demo for the automotive predictive maintenance use case.

This script keeps everything local/offline and shows how a Master Agent
coordinates Worker Agents (analysis, diagnosis, engagement, scheduling,
feedback, manufacturing insights, and UEBA security) using the existing
prediction models in `prediction_engine.py`.

Run:
    python agents_orchestrator.py
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

from prediction_engine import run_full_prediction, generate_explanation


# ----------------------------
# Worker Agents
# ----------------------------
class SecurityUebaAgent:
    """Very light UEBA guardrail for demo purposes."""

    def __init__(self):
        self.allowed_actions = {
            "ingest_telematics",
            "run_prediction",
            "propose_schedule",
            "send_customer_msg",
            "collect_feedback",
            "push_manufacturing_insights",
        }
        self.audit_log: List[str] = []

    def audit(self, action: str, meta: Dict) -> bool:
        """Return True if action is allowed; log and flag unexpected actions."""
        entry = f"[UEBA] action={action} meta={meta}"
        if action not in self.allowed_actions:
            entry += " -> BLOCKED (unauthorized action)"
            self.audit_log.append(entry)
            return False

        # Simple anomaly heuristic: block if trust_score exists but is suspiciously low
        ts = meta.get("trust_score")
        if ts is not None and ts < 5:
            entry += " -> BLOCKED (anomalous trust score)"
            self.audit_log.append(entry)
            return False

        entry += " -> OK"
        self.audit_log.append(entry)
        return True


class DataAnalysisAgent:
    """Pulls latest telematics snapshot for a vehicle."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_snapshot(self, vehicle_id: str) -> Dict:
        subset = self.df[self.df["vehicle_id"] == vehicle_id]
        if subset.empty:
            raise ValueError(f"No data for vehicle_id={vehicle_id}")
        row = subset.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
        payload = {
            "vehicle_id": row["vehicle_id"],
            "engine_temp": float(row["engine_temp"]),
            "vibration_level": float(row["vibration_level"]),
            "brake_temp": float(row["brake_temp"]),
            "speed": float(row["speed"]),
            "acceleration": float(row["acceleration"]),
            "harsh_braking_count": int(row["harsh_braking_count"]),
            "mileage": float(row["mileage"]),
            "aggressiveness_score": float(row["aggressiveness_score"]),
        }
        return payload


class DiagnosisAgent:
    """Runs ML models and crafts human-readable explanation."""

    def diagnose(self, snapshot: Dict) -> Tuple[Dict, List[str]]:
        result = run_full_prediction(snapshot)
        explanation = generate_explanation(result, snapshot)
        return result, explanation


class CustomerEngagementAgent:
    """Prepares outbound voice/message content. (No external calls here.)"""

    def craft_message(self, customer: str, vehicle: str, explanation: List[str]) -> str:
        headline = f"Hi {customer}, this is your service assistant for {vehicle}."
        body = " ".join(explanation[:3])
        cta = "Press 1 to confirm service, press 2 to reschedule."
        return f"{headline} {body} {cta}"


class SchedulingAgent:
    """Matches predicted urgency with mock service center capacity."""

    def __init__(self):
        self.slots = [
            ("SC-001", "Tomorrow 09:00"),
            ("SC-001", "Tomorrow 14:00"),
            ("SC-002", "In 2 days 10:30"),
            ("SC-003", "In 3 days 16:00"),
        ]

    def propose_slot(self, urgency: str) -> Dict:
        # High urgency -> earliest slot; otherwise pick a later one
        if urgency == "high":
            site, slot = self.slots[0]
        elif urgency == "medium":
            site, slot = self.slots[1]
        else:
            site, slot = random.choice(self.slots[2:])
        return {"service_center": site, "slot": slot}


class FeedbackAgent:
    """Collects post-service signal (simulated)."""

    def collect_feedback(self, vehicle_id: str) -> Dict:
        rating = random.choice([4, 5])  # optimistic for demo
        comment = "Service completed; vehicle feels smoother now."
        return {"vehicle_id": vehicle_id, "rating": rating, "comment": comment}


class ManufacturingInsightsAgent:
    """Maps predicted issues to CAPA/RCA-like suggestions."""

    def __init__(self):
        self.capa_rules = {
            "engine_temp": "Check cooling fan relay batch; validate firmware PID loop.",
            "brake_temp": "Inspect brake pad material supplier lot; calibrate ABS thresholds.",
            "vibration_level": "Verify engine mount torque specs; inspect driveshaft balance.",
            "harsh_braking_count": "Review ADAS braking calibration; driver coaching content.",
        }

    def generate(self, snapshot: Dict, explanation: List[str]) -> List[str]:
        insights = []
        for key, rec in self.capa_rules.items():
            if key.replace("_", " ") in " ".join(explanation):
                insights.append(f"{key}: {rec}")
        if not insights:
            insights.append("No recurring CAPA match; monitor next 2 drive cycles.")
        return insights


# ----------------------------
# Master Agent
# ----------------------------
@dataclass
class MasterAgent:
    ueba: SecurityUebaAgent
    data_agent: DataAnalysisAgent
    diagnosis_agent: DiagnosisAgent
    engagement_agent: CustomerEngagementAgent
    scheduling_agent: SchedulingAgent
    feedback_agent: FeedbackAgent
    manufacturing_agent: ManufacturingInsightsAgent
    transcript: List[str] = field(default_factory=list)

    def log(self, message: str):
        print(message)
        self.transcript.append(message)

    def run_pipeline(self, vehicle_id: str, customer_name: str):
        self.log(f"\n[M] Starting workflow for vehicle {vehicle_id}")

        # 1) Ingest telematics
        if not self.ueba.audit("ingest_telematics", {"vehicle_id": vehicle_id}):
            self.log("UEBA blocked telematics ingest.")
            return
        snapshot = self.data_agent.get_snapshot(vehicle_id)
        self.log(f"[Data] Snapshot: {snapshot}")

        # 2) Diagnosis
        if not self.ueba.audit("run_prediction", {"vehicle_id": vehicle_id}):
            self.log("UEBA blocked prediction.")
            return
        prediction, explanation = self.diagnosis_agent.diagnose(snapshot)
        self.log(f"[Diag] Prediction: {prediction}")
        self.log(f"[Diag] Explanation bullets: {explanation}")

        # 3) Decide urgency
        trust = prediction.get("trust_score", 0)
        urgency = "high" if trust >= 80 else "medium" if trust >= 50 else "low"

        # 4) Customer outreach
        if self.ueba.audit("send_customer_msg", {"vehicle_id": vehicle_id, "trust_score": trust}):
            msg = self.engagement_agent.craft_message(
                customer=customer_name,
                vehicle=vehicle_id,
                explanation=explanation,
            )
            self.log(f"[Engage] Outbound IVR/text: {msg}")
        else:
            self.log("[Engage] Blocked by UEBA.")

        # 5) Scheduling
        if self.ueba.audit("propose_schedule", {"vehicle_id": vehicle_id, "urgency": urgency}):
            slot = self.scheduling_agent.propose_slot(urgency)
            self.log(f"[Schedule] Proposed slot: {slot}")

        # 6) Feedback capture
        if self.ueba.audit("collect_feedback", {"vehicle_id": vehicle_id}):
            feedback = self.feedback_agent.collect_feedback(vehicle_id)
            self.log(f"[Feedback] {feedback}")

        # 7) Manufacturing insights
        if self.ueba.audit("push_manufacturing_insights", {"vehicle_id": vehicle_id}):
            insights = self.manufacturing_agent.generate(snapshot, explanation)
            self.log(f"[MFG] CAPA/RCA suggestions: {insights}")

        # 8) Final UEBA log
        self.log("\n[UEBA Audit Trail]")
        for entry in self.ueba.audit_log:
            self.log(entry)


# ----------------------------
# Demo entrypoint
# ----------------------------
def load_demo_df() -> pd.DataFrame:
    try:
        return pd.read_csv("vehicle_telematics_processed.csv")
    except FileNotFoundError:
        # Fall back to raw demo data if processed not present
        return pd.read_csv("vehicle_telematics_demo.csv")


def run_demo():
    df = load_demo_df()
    vehicle_id = random.choice(df["vehicle_id"].unique().tolist())

    master = MasterAgent(
        ueba=SecurityUebaAgent(),
        data_agent=DataAnalysisAgent(df),
        diagnosis_agent=DiagnosisAgent(),
        engagement_agent=CustomerEngagementAgent(),
        scheduling_agent=SchedulingAgent(),
        feedback_agent=FeedbackAgent(),
        manufacturing_agent=ManufacturingInsightsAgent(),
    )
    master.run_pipeline(vehicle_id=vehicle_id, customer_name="Asha")


if __name__ == "__main__":
    run_demo()

