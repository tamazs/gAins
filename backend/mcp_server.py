"""
MCP server exposing the three gAins agent tools over stdio transport.
Spawned as a subprocess by GainsAgent; communicates via stdin/stdout.
"""

import os
import sys

# Ensure backend/ is importable when run as a subprocess from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP
from tools.embedder import embed_text
from tools.mongo_vector_store import similarity_search
from tools.mongo_session_store import get_recent_sessions, get_goal, get_goal_entries

mcp = FastMCP("gAins-tools")


@mcp.tool()
def rag_tool(query: str) -> str:
    """
    Search the training science knowledge base for information relevant to the query.
    Use this whenever you need evidence-based guidance on programming, periodisation,
    exercise selection, rep ranges, recovery, or nutrition.
    """
    query_embedding = embed_text(query)
    results = similarity_search(query_embedding, top_k=3)

    if not results:
        return "No relevant documents found."

    sections = []
    for text, score, source in results:
        sections.append(f"[Source: {source} | similarity: {score:.2f}]\n{text}")

    return "\n\n---\n\n".join(sections)


@mcp.tool()
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


@mcp.tool()
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
        "Entry history (most recent first):",
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


if __name__ == "__main__":
    mcp.run()
