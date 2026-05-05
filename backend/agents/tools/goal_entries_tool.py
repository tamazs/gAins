from langchain_core.tools import tool

from tools.mongo_session_store import get_goal, get_goal_entries


@tool
def goal_entries_tool(user_id: str) -> str:
    """
    Retrieve the history of training entries logged toward the user's active goal.
    Use this to identify progression trends (weight, reps, RPE over time) and
    inform your next-session recommendation.
    """
    goal = get_goal(user_id)
    if not goal:
        return f"No active goal found for user {user_id}."

    exercise_name = goal.get("exercise_name", "unknown")
    entries = get_goal_entries(user_id, exercise_name)

    if not entries:
        return (
            f"Goal is set for {exercise_name} "
            f"(target: {goal['target_reps']} reps @ {goal['target_weight_kg']}kg) "
            f"but no entries have been logged yet."
        )

    lines = [
        f"Goal: {exercise_name} — target {goal['target_reps']} rep(s) @ {goal['target_weight_kg']}kg",
        f"Entry history (most recent first):",
        "",
    ]
    for entry in entries:
        date = entry.get("date", "unknown date")
        sets = entry.get("sets", [])
        sets_text = ", ".join(
            f"{s['reps']} reps @ {s['weight_kg']}kg" + (f" RPE {s['rpe']}" if s.get("rpe") else "")
            for s in sets
        )
        notes = entry.get("notes", "")
        lines.append(f"  {date}: {sets_text}" + (f" | {notes}" if notes else ""))

    return "\n".join(lines)
