"""                                                                                                                                                                                                                                                             
CrewAI-flavored orchestration demo for the automotive predictive maintenance use case.

Default behavior:
- Builds CrewAI agents/tasks (if the library is installed) for clarity.
- Runs a DRY-RUN execution using local Python functions to avoid LLM/API calls.

To run with real CrewAI + an LLM:
    1) pip install crewai langchain-openai
    2) set OPENAI_API_KEY (OpenAI) or GOOGLE_API_KEY (Gemini)
    3) set USE_CREW_KICKOFF=1
    4) python agents_orchestrator_crew.py

Environment loading:
- .env is loaded automatically (dotenv), so you can store GOOGLE_API_KEY / OPENAI_API_KEY there.
"""
import os
import random
import signal
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

from prediction_engine import run_full_prediction, generate_explanation

# Force UTF-8 console to avoid cp1252 issues with rich/emoji on Windows.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Windows compatibility: CrewAI references POSIX signals missing on Windows.
_fallback_sig = getattr(signal, "SIGTERM", 1)
for _sig in ["SIGHUP", "SIGTSTP", "SIGUSR1", "SIGUSR2", "SIGQUIT", "SIGCONT"]:
    if not hasattr(signal, _sig):
        setattr(signal, _sig, _fallback_sig)

# Load environment (supports GOOGLE_API_KEY for Gemini or OPENAI_API_KEY)
load_dotenv()

# Optional: CrewAI imports
try:
    from crewai import Agent, Crew, Process, Task

    CREW_AVAILABLE = True
except Exception:
    CREW_AVAILABLE = False

# Optional: Gemini via LangChain
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ---------------------------------------
# Shared business logic (no external APIs)
# ---------------------------------------
def load_demo_df() -> pd.DataFrame:
    try:
        return pd.read_csv("vehicle_telematics_processed.csv")
    except FileNotFoundError:
        return pd.read_csv("vehicle_telematics_demo.csv")


def get_snapshot(df: pd.DataFrame, vehicle_id: str) -> Dict:
    subset = df[df["vehicle_id"] == vehicle_id]
    if subset.empty:
        raise ValueError(f"No data for vehicle_id={vehicle_id}")
    row = subset.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
    return {
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


def diagnose(snapshot: Dict) -> Tuple[Dict, List[str]]:
    result = run_full_prediction(snapshot)
    explanation = generate_explanation(result, snapshot)
    return result, explanation


def craft_message(customer: str, vehicle: str, explanation: List[str]) -> str:
    headline = f"Hi {customer}, this is your service assistant for {vehicle}."
    body = " ".join(explanation[:3])
    cta = "Press 1 to confirm service, press 2 to reschedule."
    return f"{headline} {body} {cta}"


def propose_slot(urgency: str) -> Dict:
    slots = [
        ("SC-001", "Tomorrow 09:00"),
        ("SC-001", "Tomorrow 14:00"),
        ("SC-002", "In 2 days 10:30"),
        ("SC-003", "In 3 days 16:00"),
    ]
    if urgency == "high":
        site, slot = slots[0]
    elif urgency == "medium":
        site, slot = slots[1]
    else:
        site, slot = random.choice(slots[2:])
    return {"service_center": site, "slot": slot}


def collect_feedback(vehicle_id: str) -> Dict:
    return {
        "vehicle_id": vehicle_id,
        "rating": random.choice([4, 5]),
        "comment": "Service completed; vehicle feels smoother now.",
    }


def manufacturing_insights(explanation: List[str]) -> List[str]:
    capa_rules = {
        "engine temp": "Check cooling fan relay batch; validate firmware PID loop.",
        "brake temp": "Inspect brake pad supplier lot; calibrate ABS thresholds.",
        "vibration": "Verify engine mount torque specs; inspect driveshaft balance.",
        "braking": "Review ADAS braking calibration; driver coaching content.",
    }
    exp_text = " ".join(explanation).lower()
    hits = [rec for key, rec in capa_rules.items() if key in exp_text]
    return hits or ["No recurring CAPA match; monitor next 2 drive cycles."]


# ---------------------------------------
# CrewAI config (optional)
# ---------------------------------------
def build_crew(df: pd.DataFrame, vehicle_id: str, customer: str = "Asha", llm=None):
    if not CREW_AVAILABLE:
        raise RuntimeError("CrewAI not installed. Install with: pip install crewai")

    # Agents
    data_agent = Agent(
        role="Data Analysis Agent",
        goal="Fetch latest telematics snapshot for the requested vehicle.",
        backstory="Continuously monitors vehicle streams and prepares clean payloads.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    diag_agent = Agent(
        role="Diagnosis Agent",
        goal="Run ML models to predict failures and craft concise explanations.",
        backstory="Uses existing predictive maintenance models and heuristics.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    engage_agent = Agent(
        role="Customer Engagement Agent",
        goal="Draft persuasive IVR-ready message for the customer.",
        backstory="Keeps it under 70 words and simple.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    scheduling_agent = Agent(
        role="Scheduling Agent",
        goal="Propose the best appointment slot based on urgency.",
        backstory="Knows service center capacity and customer preference rules.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    feedback_agent = Agent(
        role="Feedback Agent",
        goal="Capture post-service sentiment for model reinforcement.",
        backstory="Returns short rating/comment payloads.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    mfg_agent = Agent(
        role="Manufacturing Insights Agent",
        goal="Map predicted issues to CAPA/RCA suggestions for engineering.",
        backstory="Looks for recurring patterns and recommends fixes.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # Tasks (descriptions are what the LLM would follow; we also supply local execs)
    t1 = Task(
        description=f"Pull a fresh telematics snapshot for vehicle {vehicle_id}.",
        agent=data_agent,
        async_execution=False,
        expected_output="JSON with sensor readings and usage metrics.",
    )
    t2 = Task(
        description="Run the predictive maintenance models and produce bullet explanations.",
        agent=diag_agent,
        expected_output="Prediction JSON + 3-5 bullet explanations.",
    )
    t3 = Task(
        description=f"Draft a concise IVR message for {customer} referencing the findings.",
        agent=engage_agent,
        expected_output="<=70 word IVR script with CTA.",
    )
    t4 = Task(
        description="Propose a service slot based on urgency level.",
        agent=scheduling_agent,
        expected_output="Service center ID and slot time.",
    )
    t5 = Task(
        description="Simulate post-service feedback capture.",
        agent=feedback_agent,
        expected_output="Rating (1-5) and brief comment.",
    )
    t6 = Task(
        description="Generate CAPA/RCA suggestions for manufacturing.",
        agent=mfg_agent,
        expected_output="2-3 CAPA/RCA bullet points.",
    )

    crew = Crew(
        agents=[data_agent, diag_agent, engage_agent, scheduling_agent, feedback_agent, mfg_agent],
        tasks=[t1, t2, t3, t4, t5, t6],
        process=Process.sequential,
        verbose=True,
    )
    return crew


# ---------------------------------------
# DRY-RUN pipeline (runs without CrewAI)
# ---------------------------------------
def run_dry_pipeline(df: pd.DataFrame, vehicle_id: str, customer: str = "Asha"):
    print("\n[DRY-RUN] Crew-style orchestration (no LLM/API calls).")
    snapshot = get_snapshot(df, vehicle_id)
    print(f"[Task 1] Data snapshot -> {snapshot}")

    prediction, explanation = diagnose(snapshot)
    print(f"[Task 2] Prediction -> {prediction}")
    print(f"[Task 2] Explanation -> {explanation}")

    trust = prediction.get("trust_score", 0)
    urgency = "high" if trust >= 80 else "medium" if trust >= 50 else "low"

    msg = craft_message(customer=customer, vehicle=vehicle_id, explanation=explanation)
    print(f"[Task 3] IVR message -> {msg}")

    slot = propose_slot(urgency)
    print(f"[Task 4] Proposed slot -> {slot}")

    fb = collect_feedback(vehicle_id)
    print(f"[Task 5] Feedback -> {fb}")

    capa = manufacturing_insights(explanation)
    print(f"[Task 6] CAPA/RCA suggestions -> {capa}")

    return {
        "snapshot": snapshot,
        "prediction": prediction,
        "explanation": explanation,
        "ivr_message": msg,
        "slot": slot,
        "feedback": fb,
        "capa": capa,
        "urgency": urgency,
    }


# ---------------------------------------
# Entrypoint
# ---------------------------------------
def main():
    df = load_demo_df()
    vehicle_id = random.choice(df["vehicle_id"].unique().tolist())

    use_crew = os.getenv("USE_CREW_KICKOFF") == "1"
    llm = None

    # Prefer Gemini if available and key is present
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if GEMINI_AVAILABLE and gemini_key:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        print("[INFO] Using Gemini via GOOGLE_API_KEY for CrewAI agents.")

    if CREW_AVAILABLE and use_crew:
        print("[INFO] Running with CrewAI kickoff (requires LLM credentials).")
        crew = build_crew(df, vehicle_id, llm=llm)
        try:
            result = crew.kickoff()
            print("\n[Crew Output]")
            print(result)
            return
        except Exception as e:
            print(f"[WARN] CrewAI kickoff failed ({e}). Falling back to dry-run.")
            run_dry_pipeline(df, vehicle_id)
            return
    else:
        if not CREW_AVAILABLE:
            print("[WARN] CrewAI not installed; using dry-run path.")
        else:
            print("[INFO] USE_CREW_KICKOFF not set; using dry-run path to avoid LLM calls.")
        run_dry_pipeline(df, vehicle_id)


if __name__ == "__main__":
    main()

