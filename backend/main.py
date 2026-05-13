import json
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError

from agents.gains_agent import GainsAgent
from auth import hash_password, verify_password, create_access_token
from models.user_models import RegisterRequest, LoginRequest, AuthResponse
from models.workout_models import (
    WorkoutSessionRequest, WorkoutAdviceResponse, ExerciseAdvice,
    GoalRequest, GoalResponse,
    GoalEntryRequest, GoalEntryResponse, GoalAdviceResponse,
)
from tools.mongo_session_store import (
    save_session, get_recent_sessions,
    save_goal, get_goal,
    save_goal_entry, get_goal_entries,
)
from tools.mongo_user_store import create_user, get_user_by_email

load_dotenv()

app = FastAPI()
agent = GainsAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=AuthResponse)
def register(body: RegisterRequest) -> AuthResponse:
    """Create a new account and return a JWT so the user is immediately logged in."""
    user_id = str(uuid.uuid4())
    try:
        create_user(
            user_id=user_id,
            email=body.email,
            username=body.username,
            hashed_password=hash_password(body.password),
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="An account with that email already exists.")

    return AuthResponse(
        access_token=create_access_token(user_id),
        user_id=user_id,
        username=body.username,
    )


@app.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest) -> AuthResponse:
    """Verify credentials and return a JWT."""
    user = get_user_by_email(body.email)

    dummy_hash = "$2b$12$kd2hDMp8yYSNPxO8fiay1.2gsrp08RWy7MLOu8IGqY4QHVHCtMZfm"
    password_ok = verify_password(body.password, user["hashed_password"] if user else dummy_hash)

    if not user or not password_ok:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    return AuthResponse(
        access_token=create_access_token(user["user_id"]),
        user_id=user["user_id"],
        username=user["username"],
    )


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=WorkoutAdviceResponse)
def analyse_session(session: WorkoutSessionRequest) -> WorkoutAdviceResponse:
    exercises_text = "\n".join(
        f"- {e.name} ({e.muscle_group}): "
        + ", ".join(f"{s.reps} reps @ {s.weight_kg}kg" + (f" RPE {s.rpe}" if s.rpe else "") for s in e.sets)
        for e in session.exercises
    )

    prompt = f"""Analyse this workout session for user {session.user_id}.

Date: {session.date.strftime('%Y-%m-%d')}
Notes: {session.notes or 'none'}
Exercises:
{exercises_text}

Respond with a JSON object matching this exact structure (no markdown, no extra text):
{{
  "overall_summary": "...",
  "exercise_advice": [
    {{
      "exercise_name": "...",
      "recommendation": "...",
      "reasoning": "...",
      "suggested_weight_kg": null,
      "suggested_reps": null,
      "suggested_sets": null
    }}
  ],
  "recovery_flag": false,
  "sources_used": []
}}"""

    raw, sources = agent.run(prompt)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Agent returned non-JSON: {raw}")

    session_id = str(uuid.uuid4())
    generated_at = datetime.now()
    analysis = {
        "overall_summary": data["overall_summary"],
        "exercise_advice": data["exercise_advice"],
        "recovery_flag": data.get("recovery_flag", False),
        "sources_used": sources,
        "generated_at": generated_at.isoformat(),
    }
    save_session(session_id, {**session.model_dump(mode="json"), "analysis": analysis})

    return WorkoutAdviceResponse(
        user_id=session.user_id,
        session_id=session_id,
        generated_at=generated_at,
        overall_summary=data["overall_summary"],
        exercise_advice=[ExerciseAdvice(**e) for e in data["exercise_advice"]],
        recovery_flag=data.get("recovery_flag", False),
        sources_used=sources,
    )


@app.get("/sessions/{user_id}")
def get_sessions(user_id: str, limit: int = 10) -> list:
    """Return the most recent sessions for a user (newest first)."""
    return get_recent_sessions(user_id, limit=limit)


# ---------------------------------------------------------------------------
# Goal endpoints
# ---------------------------------------------------------------------------

@app.post("/goals", response_model=GoalResponse)
def set_goal(goal: GoalRequest) -> GoalResponse:
    """Create or replace the user's active training goal."""
    goal_id = str(uuid.uuid4())
    created_at = datetime.now()

    payload = {**goal.model_dump(mode="json"), "created_at": created_at}
    save_goal(goal_id, payload)

    return GoalResponse(goal_id=goal_id, created_at=created_at, **goal.model_dump(mode="json"))


@app.get("/goals/{user_id}", response_model=GoalResponse)
def retrieve_goal(user_id: str) -> GoalResponse:
    """Retrieve the user's current active goal."""
    doc = get_goal(user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"No active goal found for user '{user_id}'")
    return GoalResponse(**doc)


@app.get("/goals/entries/{user_id}", response_model=list[GoalEntryResponse])
def get_entries_for_user(user_id: str, limit: int = 20) -> list:
    """Fetch the most recent goal entries for a user (newest first)."""
    goal = get_goal(user_id)
    if not goal:
        raise HTTPException(status_code=404, detail="No active goal found for this user.")
    exercise_name = goal["exercise_name"]
    entries = get_goal_entries(user_id, exercise_name, limit=limit)
    return [GoalEntryResponse(**e) for e in entries]


@app.post("/goals/entries", response_model=GoalEntryResponse)
def log_goal_entry(entry: GoalEntryRequest) -> GoalEntryResponse:
    """Log a training entry toward the user's active goal."""
    goal = get_goal(entry.user_id)
    if not goal:
        raise HTTPException(status_code=404, detail="No active goal found. Set a goal first.")

    entry_id = str(uuid.uuid4())
    exercise_name = goal["exercise_name"]

    payload = {**entry.model_dump(mode="json"), "exercise_name": exercise_name}
    save_goal_entry(entry_id, entry.user_id, payload)

    return GoalEntryResponse(
        entry_id=entry_id,
        exercise_name=exercise_name,
        **entry.model_dump(mode="json"),
    )


@app.post("/goals/analyse", response_model=GoalAdviceResponse)
def analyse_goal(body: dict) -> GoalAdviceResponse:
    """Get agent advice on progress toward the user's active goal."""
    user_id = body.get("user_id", "").strip()
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    goal = get_goal(user_id)
    if not goal:
        raise HTTPException(status_code=404, detail="No active goal found. Set a goal first.")

    exercise_name = goal["exercise_name"]
    target_reps = goal["target_reps"]
    target_weight = goal["target_weight_kg"]

    prompt = f"""Analyse the training progress of user {user_id} toward their goal.

Goal: {exercise_name} — {target_reps} rep(s) @ {target_weight}kg
{f"Deadline: {goal['deadline']}" if goal.get("deadline") else ""}
{f"Notes: {goal['notes']}" if goal.get("notes") else ""}

Instructions:
1. Call goal_entries_tool with user_id="{user_id}" to retrieve their entry history.
2. Call rag_tool with a query relevant to progressing in {exercise_name} toward the target.
3. Based on the entry history and retrieved evidence, give specific advice on how to reach the target.

Respond with a JSON object matching this exact structure (no markdown, no extra text):
{{
  "advice": "...",
  "next_session_suggestion": "..."
}}"""

    raw, sources = agent.run(prompt)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Agent returned non-JSON: {raw}")

    return GoalAdviceResponse(
        user_id=user_id,
        goal_exercise=exercise_name,
        target_weight_kg=target_weight,
        target_reps=target_reps,
        advice=data["advice"],
        next_session_suggestion=data["next_session_suggestion"],
        sources_used=sources,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)