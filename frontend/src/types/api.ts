// Auth
export interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  username: string;
}

// Sessions
export interface ExerciseSet {
  reps: number;
  weight_kg: number;
  rpe?: number;
}

export interface Exercise {
  name: string;
  muscle_group: string;
  sets: ExerciseSet[];
}

export interface WorkoutSessionRequest {
  user_id: string;
  date: string;
  exercises: Exercise[];
  notes?: string;
}

export interface ExerciseAdvice {
  exercise_name: string;
  recommendation: string;
  reasoning: string;
  suggested_weight_kg?: number;
  suggested_reps?: number;
  suggested_sets?: number;
}

export interface WorkoutAdviceResponse {
  user_id: string;
  session_id: string;
  generated_at: string;
  overall_summary: string;
  exercise_advice: ExerciseAdvice[];
  recovery_flag: boolean;
  sources_used: string[];
}

export interface SessionDocument extends WorkoutSessionRequest {
  session_id?: string;
}

// Goals
export interface GoalRequest {
  user_id: string;
  exercise_name: string;
  muscle_group: string;
  target_weight_kg: number;
  target_reps: number;
  deadline?: string;
  notes?: string;
}

export interface GoalResponse {
  goal_id: string;
  user_id: string;
  exercise_name: string;
  muscle_group: string;
  target_weight_kg: number;
  target_reps: number;
  deadline?: string;
  notes?: string;
  created_at: string;
}

export interface GoalEntryRequest {
  user_id: string;
  date: string;
  sets: ExerciseSet[];
  notes?: string;
}

export interface GoalEntryResponse {
  entry_id: string;
  user_id: string;
  exercise_name: string;
  date: string;
  sets: ExerciseSet[];
  notes?: string;
}

export interface GoalAdviceResponse {
  user_id: string;
  goal_exercise: string;
  target_weight_kg: number;
  target_reps: number;
  advice: string;
  next_session_suggestion: string;
  sources_used: string[];
}
