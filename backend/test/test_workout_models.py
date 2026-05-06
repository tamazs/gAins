"""
test_workout_models.py — Unit tests for models/workout_models.py

Covers every field validator on ExerciseSet, Exercise, WorkoutSessionRequest,
GoalRequest, and GoalEntryRequest.
"""

import pytest
from datetime import datetime, timedelta, timezone, date
from pydantic import ValidationError

from models.workout_models import (
    ExerciseSet,
    Exercise,
    WorkoutSessionRequest,
    GoalRequest,
    GoalEntryRequest,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _past_dt() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=1)


def _future_dt() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


# ── ExerciseSet ───────────────────────────────────────────────────────────────

class TestExerciseSet:
    def test_valid_set_passes(self):
        s = ExerciseSet(reps=5, weight_kg=100.0)
        assert s.reps == 5
        assert s.weight_kg == 100.0
        assert s.rpe is None

    def test_reps_zero_raises(self):
        with pytest.raises(ValidationError):
            ExerciseSet(reps=0, weight_kg=100.0)

    def test_reps_negative_raises(self):
        with pytest.raises(ValidationError):
            ExerciseSet(reps=-1, weight_kg=100.0)

    def test_reps_1_passes(self):
        s = ExerciseSet(reps=1, weight_kg=50.0)
        assert s.reps == 1

    def test_reps_100_passes(self):
        s = ExerciseSet(reps=100, weight_kg=50.0)
        assert s.reps == 100

    def test_reps_101_raises(self):
        with pytest.raises(ValidationError):
            ExerciseSet(reps=101, weight_kg=100.0)

    def test_weight_zero_passes(self):
        s = ExerciseSet(reps=5, weight_kg=0.0)
        assert s.weight_kg == 0.0

    def test_weight_negative_raises(self):
        with pytest.raises(ValidationError):
            ExerciseSet(reps=5, weight_kg=-0.1)

    def test_weight_500_passes(self):
        s = ExerciseSet(reps=5, weight_kg=500.0)
        assert s.weight_kg == 500.0

    def test_weight_above_500_raises(self):
        with pytest.raises(ValidationError):
            ExerciseSet(reps=5, weight_kg=500.1)

    def test_rpe_none_passes(self):
        s = ExerciseSet(reps=5, weight_kg=100.0, rpe=None)
        assert s.rpe is None

    def test_rpe_1_passes(self):
        s = ExerciseSet(reps=5, weight_kg=100.0, rpe=1.0)
        assert s.rpe == 1.0

    def test_rpe_10_passes(self):
        s = ExerciseSet(reps=5, weight_kg=100.0, rpe=10.0)
        assert s.rpe == 10.0

    def test_rpe_below_1_raises(self):
        with pytest.raises(ValidationError):
            ExerciseSet(reps=5, weight_kg=100.0, rpe=0.5)

    def test_rpe_above_10_raises(self):
        with pytest.raises(ValidationError):
            ExerciseSet(reps=5, weight_kg=100.0, rpe=10.5)


# ── Exercise ──────────────────────────────────────────────────────────────────

class TestExercise:
    def _sets(self, n: int = 1):
        return [{"reps": 5, "weight_kg": 100.0}] * n

    def test_valid_exercise_passes(self):
        ex = Exercise(name="bench press", muscle_group="chest", sets=self._sets())
        assert ex.name == "Bench Press"

    def test_name_is_title_cased(self):
        ex = Exercise(name="back squat", muscle_group="quads", sets=self._sets())
        assert ex.name == "Back Squat"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            Exercise(name="", muscle_group="chest", sets=self._sets())

    def test_whitespace_only_name_raises(self):
        with pytest.raises(ValidationError):
            Exercise(name="   ", muscle_group="chest", sets=self._sets())

    def test_invalid_muscle_group_raises(self):
        with pytest.raises(ValidationError, match="muscle_group"):
            Exercise(name="Curl", muscle_group="unknown", sets=self._sets())

    @pytest.mark.parametrize("mg", [
        "chest", "back", "shoulders", "biceps", "triceps",
        "legs", "quads", "hamstrings", "glutes", "calves", "core",
    ])
    def test_all_valid_muscle_groups_pass(self, mg):
        ex = Exercise(name="Exercise", muscle_group=mg, sets=self._sets())
        assert ex.muscle_group == mg

    def test_empty_sets_raises(self):
        with pytest.raises(ValidationError, match="at least one set"):
            Exercise(name="Squat", muscle_group="quads", sets=[])

    def test_21_sets_raises(self):
        with pytest.raises(ValidationError, match="20 sets"):
            Exercise(name="Squat", muscle_group="quads", sets=self._sets(21))

    def test_20_sets_passes(self):
        ex = Exercise(name="Squat", muscle_group="quads", sets=self._sets(20))
        assert len(ex.sets) == 20


# ── WorkoutSessionRequest ─────────────────────────────────────────────────────

class TestWorkoutSessionRequest:
    def _exercise(self, name: str = "Bench Press"):
        return {
            "name": name,
            "muscle_group": "chest",
            "sets": [{"reps": 5, "weight_kg": 100.0}],
        }

    def test_valid_session_passes(self):
        req = WorkoutSessionRequest(
            user_id="user-1",
            date=_past_dt(),
            exercises=[self._exercise()],
        )
        assert req.user_id == "user-1"

    def test_future_date_raises(self):
        with pytest.raises(ValidationError, match="[Ff]uture"):
            WorkoutSessionRequest(
                user_id="user-1",
                date=_future_dt(),
                exercises=[self._exercise()],
            )

    def test_empty_user_id_raises(self):
        with pytest.raises(ValidationError):
            WorkoutSessionRequest(
                user_id="",
                date=_past_dt(),
                exercises=[self._exercise()],
            )

    def test_whitespace_user_id_raises(self):
        with pytest.raises(ValidationError):
            WorkoutSessionRequest(
                user_id="   ",
                date=_past_dt(),
                exercises=[self._exercise()],
            )

    def test_no_exercises_raises(self):
        with pytest.raises(ValidationError, match="[Aa]t least one exercise"):
            WorkoutSessionRequest(user_id="user-1", date=_past_dt(), exercises=[])

    def test_21_exercises_raises(self):
        exercises = [self._exercise(f"Exercise {i}") for i in range(21)]
        with pytest.raises(ValidationError, match="20 exercises"):
            WorkoutSessionRequest(user_id="user-1", date=_past_dt(), exercises=exercises)

    def test_duplicate_exercise_names_raises(self):
        with pytest.raises(ValidationError, match="[Dd]uplicate"):
            WorkoutSessionRequest(
                user_id="user-1",
                date=_past_dt(),
                exercises=[self._exercise("Bench Press"), self._exercise("Bench Press")],
            )

    def test_notes_exactly_500_chars_passes(self):
        req = WorkoutSessionRequest(
            user_id="user-1",
            date=_past_dt(),
            exercises=[self._exercise()],
            notes="x" * 500,
        )
        assert len(req.notes) == 500

    def test_notes_501_chars_raises(self):
        with pytest.raises(ValidationError, match="500"):
            WorkoutSessionRequest(
                user_id="user-1",
                date=_past_dt(),
                exercises=[self._exercise()],
                notes="x" * 501,
            )

    def test_notes_none_passes(self):
        req = WorkoutSessionRequest(
            user_id="user-1",
            date=_past_dt(),
            exercises=[self._exercise()],
            notes=None,
        )
        assert req.notes is None


# ── GoalRequest ───────────────────────────────────────────────────────────────

class TestGoalRequest:
    def _valid(self, **overrides) -> dict:
        base = dict(
            user_id="user-1",
            exercise_name="Squat",
            muscle_group="quads",
            target_weight_kg=140.0,
            target_reps=1,
        )
        base.update(overrides)
        return base

    def test_valid_goal_passes(self):
        g = GoalRequest(**self._valid())
        assert g.exercise_name == "Squat"

    def test_exercise_name_title_cased(self):
        g = GoalRequest(**self._valid(exercise_name="back squat"))
        assert g.exercise_name == "Back Squat"

    def test_empty_exercise_name_raises(self):
        with pytest.raises(ValidationError):
            GoalRequest(**self._valid(exercise_name=""))

    def test_empty_user_id_raises(self):
        with pytest.raises(ValidationError):
            GoalRequest(**self._valid(user_id=""))

    def test_invalid_muscle_group_raises(self):
        with pytest.raises(ValidationError):
            GoalRequest(**self._valid(muscle_group="invalid"))

    def test_target_weight_zero_raises(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            GoalRequest(**self._valid(target_weight_kg=0))

    def test_target_weight_negative_raises(self):
        with pytest.raises(ValidationError):
            GoalRequest(**self._valid(target_weight_kg=-10.0))

    def test_target_weight_501_raises(self):
        with pytest.raises(ValidationError, match="500"):
            GoalRequest(**self._valid(target_weight_kg=501.0))

    def test_target_reps_zero_raises(self):
        with pytest.raises(ValidationError, match="at least 1"):
            GoalRequest(**self._valid(target_reps=0))

    def test_target_reps_101_raises(self):
        with pytest.raises(ValidationError, match="100"):
            GoalRequest(**self._valid(target_reps=101))

    def test_deadline_today_raises(self):
        with pytest.raises(ValidationError, match="[Ff]uture"):
            GoalRequest(**self._valid(deadline=date.today()))

    def test_deadline_in_past_raises(self):
        with pytest.raises(ValidationError):
            GoalRequest(**self._valid(deadline=date(2020, 1, 1)))

    def test_deadline_tomorrow_passes(self):
        tomorrow = date.today() + timedelta(days=1)
        g = GoalRequest(**self._valid(deadline=tomorrow))
        assert g.deadline == tomorrow

    def test_deadline_none_passes(self):
        g = GoalRequest(**self._valid(deadline=None))
        assert g.deadline is None

    def test_notes_501_chars_raises(self):
        with pytest.raises(ValidationError, match="500"):
            GoalRequest(**self._valid(notes="x" * 501))

    def test_notes_500_chars_passes(self):
        g = GoalRequest(**self._valid(notes="x" * 500))
        assert len(g.notes) == 500


# ── GoalEntryRequest ──────────────────────────────────────────────────────────

class TestGoalEntryRequest:
    def _valid(self, **overrides) -> dict:
        base = dict(
            user_id="user-1",
            date=_past_dt(),
            sets=[{"reps": 3, "weight_kg": 130.0}],
        )
        base.update(overrides)
        return base

    def test_valid_entry_passes(self):
        e = GoalEntryRequest(**self._valid())
        assert e.user_id == "user-1"

    def test_future_date_raises(self):
        with pytest.raises(ValidationError, match="[Ff]uture"):
            GoalEntryRequest(**self._valid(date=_future_dt()))

    def test_empty_user_id_raises(self):
        with pytest.raises(ValidationError):
            GoalEntryRequest(**self._valid(user_id=""))

    def test_empty_sets_raises(self):
        with pytest.raises(ValidationError, match="[Aa]t least one set"):
            GoalEntryRequest(**self._valid(sets=[]))

    def test_notes_501_chars_raises(self):
        with pytest.raises(ValidationError, match="500"):
            GoalEntryRequest(**self._valid(notes="x" * 501))

    def test_notes_none_passes(self):
        e = GoalEntryRequest(**self._valid(notes=None))
        assert e.notes is None
