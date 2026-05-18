# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**gAins** is a strength training AI assistant. Users log workouts, set training goals, and receive AI-generated coaching advice. The AI agent retrieves relevant training science from local PDFs (RAG) and the user's past sessions to give personalized recommendations.

## Commands

### Backend (run from `backend/`)

```bash
# Install dependencies
pip install -r requirements.txt

# Run dev server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest

# Run a single test file
pytest test/test_main.py -v

# Ingest RAG documents (PDFs in backend/rag_docs/)
python ingest.py

# Clear and re-ingest
python ingest.py --clear
```

### Frontend (run from `frontend/`)

```bash
npm install
npm run dev       # http://localhost:5173
npm run build
npm run lint
```

## Environment Variables

Create `backend/.env` (no `.env.example` exists):

```
MONGODB_URI=mongodb://localhost:27017
DB_NAME=gains
JWT_SECRET=<random-secret>
```

## External Dependencies

- **MongoDB** — local or remote, URI via `MONGODB_URI`
- **Ollama** — must be running locally; uses `gAinsModel` for chat and `nomic-embed-text` for embeddings

## Architecture

### Backend (`backend/`)

FastAPI app defined entirely in `main.py`. Three resource groups:

- **Auth** (`/auth/register`, `/auth/login`) — JWT (HS256) issued on login; bcrypt passwords. Logic in `auth.py`.
- **Sessions** (`/sessions`) — POST creates a session and runs the AI agent synchronously; GET retrieves past sessions by user.
- **Goals** (`/goals`, `/goals/entries`, `/goals/analyse`) — one active goal per user (upsert), entries log training attempts, analyse endpoint invokes the agent.

**Important:** No JWT middleware exists on any endpoint. The `user_id` is passed in the request body and trusted as-is. The JWT is used only as a session token on the frontend.

`GainsAgent` (`agents/gains_agent.py`) is a LangChain agent executor wrapping three tools:

| Tool | File | Purpose |
|---|---|---|
| `rag_tool` | `agents/tools/rag_tool.py` | Cosine similarity search over embedded PDF chunks |
| `session_history_tool` | `agents/tools/session_history_tool.py` | Last 5 workouts for the user |
| `goal_entries_tool` | `agents/tools/goal_entries_tool.py` | Progress entries toward the user's current goal |

There are two distinct `tools/` directories:
- `backend/tools/` — infrastructure layer: MongoDB collection accessors (`mongo_session_store.py`, `mongo_user_store.py`, `mongo_vector_store.py`), embedder, text splitter
- `backend/agents/tools/` — LangChain `@tool`-decorated wrappers that call into the infrastructure layer

MongoDB collections: `gym_sessions`, `user_goals`, `goal_entries`, `users`.

Pydantic schemas for requests/responses are in `models/`.

### Frontend (`frontend/`)

React 19 + TypeScript SPA, built with Vite, routed with React Router 7.

- **State** — Jotai atoms (`atoms/`): `authAtom` (token + user_id + username), `exercisesAtom` (cached history)
- **API** — Axios client in `api/client.ts` auto-attaches `Bearer` token. Service modules: `api/auth.ts`, `api/sessions.ts`, `api/goals.ts`
- **Pages** — `DashboardPage`, `SessionsPage`, `NewSessionPage`, `SessionDetailPage`, `GoalsPage`, plus public `LoginPage`/`RegisterPage`
- **Custom hooks** — `useAuth`, `useSessions`, `useGoal`, `useExercises` wrap API calls and atom access
- **UI primitives** — `components/ui/` (Button, Card, Input, Badge, Spinner)

All protected routes go through `components/layout/PrivateRoute`.

**Gotcha:** `SessionDetailPage` routes by array index (`/sessions/:id` where `id` is the position in the fetched array), not by `session_id`. This means links are not stable if sessions are added or removed.

### Data Models

```
User:          user_id, email (unique), username, hashed_password
Session:       session_id, user_id, date, notes, exercises[]
  Exercise:    name, muscle_group, sets[]
  Set:         reps, weight_kg, rpe (optional)
Goal:          goal_id, user_id, exercise_name, target_weight_kg, target_reps, deadline, notes
GoalEntry:     entry_id, user_id, exercise_name, date, sets[], notes
RAGDocument:   text (chunk), embedding (vector), source (PDF filename)
```

### Testing

Backend tests use pytest. `backend/test/conftest.py` stubs out Ollama/LangChain (`langchain_ollama`, `langchain_classic`) via `sys.modules` before any local imports, preventing real network calls during test collection. It also resets MongoDB collection singletons before/after each test so lazy-init mocks don't bleed between tests. Tests cover auth, Pydantic models, FastAPI endpoints (via `TestClient`), agent tool calling (mocked LLM), and Mongo store operations.
