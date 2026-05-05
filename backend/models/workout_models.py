from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional
from datetime import datetime, date, timezone

class ExerciseSet(BaseModel):
    reps: int
    weight_kg: float
    rpe: Optional[float] = None      # Rate of Perceived Exertion 1-10, optional

    @field_validator("reps")
    @classmethod
    def reps_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Reps must be at least 1")
        if v > 100:
            raise ValueError("Reps cannot exceed 100")
        return v

    @field_validator("weight_kg")
    @classmethod
    def weight_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Weight cannot be negative")
        if v > 500:
            raise ValueError("Weight exceeds realistic maximum (500kg)")
        return v

    @field_validator("rpe")
    @classmethod
    def rpe_must_be_valid_scale(cls, v):
        if v is not None and not (1.0 <= v <= 10.0):
            raise ValueError("RPE must be between 1 and 10")
        return v

class Exercise(BaseModel):
    name: str
    muscle_group: str                 # "chest", "back", "legs" etc.
    sets: List[ExerciseSet]

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Exercise name cannot be empty")
        return v.title()  # normalises "bench press" -> "Bench Press"

    @field_validator("muscle_group")
    @classmethod
    def muscle_group_must_be_valid(cls, v):
        valid = {
            "chest", "back", "shoulders", "biceps", "triceps",
            "legs", "quads", "hamstrings", "glutes", "calves", "core"
        }
        v = v.strip().lower()
        if v not in valid:
            raise ValueError(f"muscle_group must be one of: {', '.join(sorted(valid))}")
        return v

    @field_validator("sets")
    @classmethod
    def must_have_at_least_one_set(cls, v):
        if not v:
            raise ValueError("Exercise must have at least one set")
        if len(v) > 20:
            raise ValueError("Cannot log more than 20 sets per exercise")
        return v

class WorkoutSessionRequest(BaseModel):
    user_id: str
    date: datetime
    exercises: List[Exercise]
    notes: Optional[str] = None      # user can add "felt tired today" etc.

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("user_id cannot be empty")
        return v

    @field_validator("exercises")
    @classmethod
    def must_have_at_least_one_exercise(cls, v):
        if not v:
            raise ValueError("Session must contain at least one exercise")
        if len(v) > 20:
            raise ValueError("Cannot log more than 20 exercises per session")
        return v

    @field_validator("date")
    @classmethod
    def date_cannot_be_future(cls, v):
        if v > datetime.now(timezone.utc):
            raise ValueError("Session date cannot be in the future")
        return v

    @field_validator("notes")
    @classmethod
    def notes_length(cls, v):
        if v is not None and len(v) > 500:
            raise ValueError("Notes cannot exceed 500 characters")
        return v

    @model_validator(mode="after")
    def check_no_duplicate_exercises(self):
        names = [e.name.lower() for e in self.exercises]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate exercises in session — combine them into one entry")
        return self

# --- Response models ---

class ExerciseAdvice(BaseModel):
    exercise_name: str
    recommendation: str              # "Increase to 85kg next session"
    reasoning: str                   # "You completed all sets with 2+ RIR"
    suggested_weight_kg: Optional[float] = None
    suggested_reps: Optional[int] = None
    suggested_sets: Optional[int] = None

class WorkoutAdviceResponse(BaseModel):
    user_id: str
    session_id: str                  # links back to the session in MongoDB
    generated_at: datetime
    overall_summary: str             # general overview of the session
    exercise_advice: List[ExerciseAdvice]  # one per exercise
    recovery_flag: bool              # True if agent detected overtraining risk
    sources_used: List[str]          # which RAG docs were retrieved


# --- Goal models ---

class GoalRequest(BaseModel):
    user_id: str
    exercise_name: str               # e.g. "Squat"
    muscle_group: str                # validated against the same set as Exercise
    target_weight_kg: float          # the weight they want to hit
    target_reps: int                 # at that weight
    deadline: Optional[date] = None  # optional target date, e.g. "2026-08-01"
    notes: Optional[str] = None      # e.g. "preparing for a powerlifting meet"

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("user_id cannot be empty")
        return v

    @field_validator("exercise_name")
    @classmethod
    def exercise_name_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Exercise name cannot be empty")
        return v.title()

    @field_validator("muscle_group")
    @classmethod
    def muscle_group_must_be_valid(cls, v):
        valid = {
            "chest", "back", "shoulders", "biceps", "triceps",
            "legs", "quads", "hamstrings", "glutes", "calves", "core"
        }
        v = v.strip().lower()
        if v not in valid:
            raise ValueError(f"muscle_group must be one of: {', '.join(sorted(valid))}")
        return v

    @field_validator("target_weight_kg")
    @classmethod
    def target_weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Target weight must be greater than 0")
        if v > 500:
            raise ValueError("Target weight exceeds realistic maximum (500 kg)")
        return v

    @field_validator("target_reps")
    @classmethod
    def target_reps_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Target reps must be at least 1")
        if v > 100:
            raise ValueError("Target reps cannot exceed 100")
        return v

    @field_validator("deadline")
    @classmethod
    def deadline_must_be_future(cls, v):
        if v is not None and v <= date.today():
            raise ValueError("Deadline must be a future date")
        return v

    @field_validator("notes")
    @classmethod
    def notes_length(cls, v):
        if v is not None and len(v) > 500:
            raise ValueError("Notes cannot exceed 500 characters")
        return v


class GoalResponse(BaseModel):
    goal_id: str
    user_id: str
    exercise_name: str
    muscle_group: str
    target_weight_kg: float
    target_reps: int
    deadline: Optional[date] = None
    notes: Optional[str] = None
    created_at: datetime


class GoalEntryRequest(BaseModel):
    """A single training entry logged toward the user's active goal."""
    user_id: str
    date: datetime
    sets: List[ExerciseSet]          # reuses ExerciseSet — reps, weight_kg, optional RPE
    notes: Optional[str] = None

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("user_id cannot be empty")
        return v

    @field_validator("sets")
    @classmethod
    def must_have_at_least_one_set(cls, v):
        if not v:
            raise ValueError("Entry must have at least one set")
        return v

    @field_validator("date")
    @classmethod
    def date_cannot_be_future(cls, v):
        if v > datetime.now(timezone.utc):
            raise ValueError("Entry date cannot be in the future")
        return v

    @field_validator("notes")
    @classmethod
    def notes_length(cls, v):
        if v is not None and len(v) > 500:
            raise ValueError("Notes cannot exceed 500 characters")
        return v


class GoalEntryResponse(BaseModel):
    entry_id: str
    user_id: str
    exercise_name: str               # copied from active goal at time of logging
    date: datetime
    sets: List[ExerciseSet]
    notes: Optional[str] = None


class GoalAdviceResponse(BaseModel):
    user_id: str
    goal_exercise: str
    target_weight_kg: float
    target_reps: int
    advice: str                      # main progression advice
    next_session_suggestion: str     # one concrete actionable next step
    sources_used: List[str]