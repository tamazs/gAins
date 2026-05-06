import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, ArrowLeft, Sparkles } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { useSessions } from '../hooks/useSessions';
import { useExercises } from '../hooks/useExercises';
import { ExerciseFormRow } from '../components/sessions/ExerciseFormRow';
import type { ExerciseRow } from '../components/sessions/ExerciseFormRow';
import { Button } from '../components/ui/Button';
import { Textarea } from '../components/ui/Input';
import type { WorkoutAdviceResponse } from '../types/api';

let rowCounter = 0;
const makeRow = (): ExerciseRow => ({
  id: String(rowCounter++),
  exerciseName: '',
  muscleGroup: '',
  sets: [{ reps: '', weight_kg: '', rpe: '' }],
});

export function NewSessionPage() {
  const { auth } = useAuth();
  const { submitSession, loading } = useSessions();
  const { loadInitial } = useExercises();
  const navigate = useNavigate();

  const [rows, setRows] = useState<ExerciseRow[]>([makeRow()]);
  const [notes, setNotes] = useState('');
  const [error, setError] = useState('');
  const [advice, setAdvice] = useState<WorkoutAdviceResponse | null>(null);

  useEffect(() => { loadInitial(); }, []);

  const updateRow = (id: string, updated: ExerciseRow) => {
    setRows((prev) => prev.map((r) => (r.id === id ? updated : r)));
  };

  const addRow = () => setRows((prev) => [...prev, makeRow()]);
  const removeRow = (id: string) => setRows((prev) => prev.filter((r) => r.id !== id));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    // Validate
    for (const row of rows) {
      if (!row.exerciseName.trim()) {
        setError('Please select an exercise for each row.');
        return;
      }
      for (const set of row.sets) {
        if (!set.reps || !set.weight_kg) {
          setError('Please fill in reps and weight for every set.');
          return;
        }
      }
    }

    const payload = {
      user_id: auth.userId!,
      date: new Date().toISOString(),
      notes: notes.trim() || undefined,
      exercises: rows.map((row) => ({
        name: row.exerciseName,
        muscle_group: row.muscleGroup,
        sets: row.sets.map((s) => ({
          reps: parseInt(s.reps, 10),
          weight_kg: parseFloat(s.weight_kg),
          rpe: s.rpe ? parseFloat(s.rpe) : undefined,
        })),
      })),
    };

    try {
      const result = await submitSession(payload);
      setAdvice(result);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Submission failed.');
    }
  };

  // Show advice panel after submission
  if (advice) {
    return (
      <div className="space-y-6">
        <button
          onClick={() => navigate('/sessions')}
          className="flex items-center gap-2 text-sm text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
        >
          <ArrowLeft size={15} /> Back to sessions
        </button>

        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[var(--accent-muted)] border border-[var(--accent-border)] flex items-center justify-center">
            <Sparkles size={16} className="text-[var(--accent-hover)]" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-[var(--text-primary)]">AI Advice</h1>
            <p className="text-sm text-[var(--text-muted)]">Session logged — here's your personalised feedback</p>
          </div>
        </div>

        {/* Overall Summary */}
        <div className="rounded-xl border border-[var(--accent-border)] bg-[var(--accent-muted)] p-5">
          <p className="text-sm font-medium text-[var(--accent-hover)] mb-2">Overall Summary</p>
          <p className="text-sm text-[var(--text-primary)] leading-relaxed">{advice.overall_summary}</p>
        </div>

        {advice.recovery_flag && (
          <div className="rounded-xl border border-[var(--danger)]/30 bg-[var(--danger-muted)] p-4">
            <p className="text-sm font-medium text-[var(--danger)]">⚠️ Recovery Warning</p>
            <p className="text-sm text-[var(--text-secondary)] mt-1">Signs of potential overtraining detected. Consider additional rest.</p>
          </div>
        )}

        {/* Per-exercise advice */}
        <div className="space-y-4">
          {advice.exercise_advice.map((ea) => (
            <div key={ea.exercise_name} className="rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] p-5 space-y-3">
              <h3 className="font-semibold text-[var(--text-primary)]">{ea.exercise_name}</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-[var(--text-muted)] mb-1">Recommendation</p>
                  <p className="text-sm text-[var(--text-primary)]">{ea.recommendation}</p>
                </div>
                <div>
                  <p className="text-xs text-[var(--text-muted)] mb-1">Reasoning</p>
                  <p className="text-sm text-[var(--text-secondary)]">{ea.reasoning}</p>
                </div>
              </div>
              {(ea.suggested_weight_kg || ea.suggested_reps || ea.suggested_sets) && (
                <div className="flex flex-wrap gap-3 pt-2 border-t border-[var(--border)]">
                  {ea.suggested_weight_kg && (
                    <div className="text-center">
                      <p className="text-xs text-[var(--text-muted)]">Next weight</p>
                      <p className="text-lg font-bold text-[var(--accent-hover)]">{ea.suggested_weight_kg}kg</p>
                    </div>
                  )}
                  {ea.suggested_reps && (
                    <div className="text-center">
                      <p className="text-xs text-[var(--text-muted)]">Target reps</p>
                      <p className="text-lg font-bold text-[var(--accent-hover)]">{ea.suggested_reps}</p>
                    </div>
                  )}
                  {ea.suggested_sets && (
                    <div className="text-center">
                      <p className="text-xs text-[var(--text-muted)]">Sets</p>
                      <p className="text-lg font-bold text-[var(--accent-hover)]">{ea.suggested_sets}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        <Button variant="secondary" onClick={() => navigate('/sessions')}>
          View all sessions
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <button
        onClick={() => navigate('/sessions')}
        className="flex items-center gap-2 text-sm text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
      >
        <ArrowLeft size={15} /> Back
      </button>

      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Log Session</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          {new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' })}
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Exercises */}
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider">Exercises</h2>
          {rows.map((row) => (
            <ExerciseFormRow
              key={row.id}
              row={row}
              onChange={(updated) => updateRow(row.id, updated)}
              onRemove={() => removeRow(row.id)}
              showRemove={rows.length > 1}
            />
          ))}
          <Button type="button" variant="ghost" onClick={addRow}>
            <Plus size={15} /> Add exercise
          </Button>
        </div>

        {/* Notes */}
        <Textarea
          label="Session notes (optional)"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="How did the session feel? Any PRs?"
          rows={3}
        />

        {error && (
          <p className="text-sm text-[var(--danger)] bg-[var(--danger-muted)] px-3 py-2 rounded-lg border border-[var(--danger)]/20">
            {error}
          </p>
        )}

        <div className="flex gap-3">
          <Button type="submit" variant="primary" size="lg" loading={loading}>
            <Sparkles size={15} /> Analyse Session
          </Button>
        </div>
      </form>
    </div>
  );
}
