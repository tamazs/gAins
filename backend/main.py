import json
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from agents.gains_agent import GainsAgent
from models.workout_models import WorkoutSessionRequest, WorkoutAdviceResponse, ExerciseAdvice
from tools.mongo_session_store import save_session

load_dotenv()

app = FastAPI()
agent = GainsAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/test", response_model=WorkoutAdviceResponse)
def test_agent(session: WorkoutSessionRequest) -> WorkoutAdviceResponse:
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
    save_session(session_id, session.model_dump())

    return WorkoutAdviceResponse(
        user_id=session.user_id,
        session_id=session_id,
        generated_at=datetime.now(),
        overall_summary=data["overall_summary"],
        exercise_advice=[ExerciseAdvice(**e) for e in data["exercise_advice"]],
        recovery_flag=data.get("recovery_flag", False),
        sources_used=sources,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)