from langchain_core.tools import tool

from tools.mongo_session_store import get_recent_sessions


@tool
def session_history_tool(user_id: str) -> str:
    """
    Retrieve the last 5 workout sessions for a user from the database.
    Use this to compare the current session against recent history and identify
    progression trends, stalls, or recovery patterns.
    """
    sessions = get_recent_sessions(user_id, limit=5)

    if not sessions:
        return f"No previous sessions found for user {user_id}."

    lines = []
    for s in sessions:
        date = s.get("date", "unknown date")
        notes = s.get("notes", "")
        exercises = s.get("exercises", [])

        lines.append(f"Session: {date}" + (f" | Notes: {notes}" if notes else ""))
        for ex in exercises:
            sets_text = ", ".join(
                f"{st['reps']} reps @ {st['weight_kg']}kg" + (f" RPE {st['rpe']}" if st.get("rpe") else "")
                for st in ex.get("sets", [])
            )
            lines.append(f"  - {ex['name']} ({ex['muscle_group']}): {sets_text}")
        lines.append("")

    return "\n".join(lines)
