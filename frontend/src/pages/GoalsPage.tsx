import { useState, useEffect, useRef } from 'react';
import { Plus, Trash2, Sparkles, Target, ChevronDown } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { useGoal } from '../hooks/useGoal';
import { useExercises } from '../hooks/useExercises';
import { GoalCard } from '../components/goals/GoalCard';
import { GoalEntryRow } from '../components/goals/GoalEntryRow';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Card } from '../components/ui/Card';
import { PageLoader } from '../components/ui/Spinner';
import { toMuscleGroup } from '../utils/muscleGroupMap';
import type { ExerciseDbEntry } from '../types/exercise';

interface SetRow { reps: string; weight_kg: string; rpe: string; }

export function GoalsPage() {
  const { auth } = useAuth();
  const { goal, entries, advice, loading, fetchGoal, fetchEntries, createGoal, addEntry, getAdvice } = useGoal();
  const { loadInitial, search } = useExercises();

  const [initialized, setInitialized] = useState(false);
  const [showGoalForm, setShowGoalForm] = useState(false);
  const [showEntryForm, setShowEntryForm] = useState(false);
  const [error, setError] = useState('');

  // Goal form state
  const [goalQuery, setGoalQuery] = useState('');
  const [goalDropOpen, setGoalDropOpen] = useState(false);
  const [goalResults, setGoalResults] = useState<ExerciseDbEntry[]>([]);
  const [selectedExercise, setSelectedExercise] = useState<ExerciseDbEntry | null>(null);
  const [targetWeight, setTargetWeight] = useState('');
  const [targetReps, setTargetReps] = useState('');
  const [deadline, setDeadline] = useState('');
  const [goalNotes, setGoalNotes] = useState('');
  const dropRef = useRef<HTMLDivElement>(null);

  // Entry form state
  const [entrySets, setEntrySets] = useState<SetRow[]>([{ reps: '', weight_kg: '', rpe: '' }]);
  const [entryNotes, setEntryNotes] = useState('');

  useEffect(() => {
    if (auth.userId) {
      Promise.all([fetchGoal(auth.userId), fetchEntries(auth.userId), loadInitial()])
        .finally(() => setInitialized(true));
    }
  }, [auth.userId]);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropRef.current && !dropRef.current.contains(e.target as Node)) setGoalDropOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleGoalSearch = async (q: string) => {
    setGoalQuery(q);
    setGoalDropOpen(true);
    if (q.length < 2) { setGoalResults([]); return; }
    const res = await search(q);
    setGoalResults(res.slice(0, 10));
  };

  const handleSetGoal = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (!selectedExercise) { setError('Please select an exercise.'); return; }
    if (!targetWeight || !targetReps) { setError('Please fill in target weight and reps.'); return; }
    try {
      await createGoal({
        user_id: auth.userId!,
        exercise_name: selectedExercise.name,
        muscle_group: toMuscleGroup(selectedExercise.bodyPart, selectedExercise.target),
        target_weight_kg: parseFloat(targetWeight),
        target_reps: parseInt(targetReps, 10),
        deadline: deadline || undefined,
        notes: goalNotes || undefined,
      });
      setShowGoalForm(false);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to set goal');
    }
  };

  const handleLogEntry = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    for (const s of entrySets) {
      if (!s.reps || !s.weight_kg) { setError('Please fill in reps and weight for every set.'); return; }
    }
    try {
      await addEntry({
        user_id: auth.userId!,
        date: new Date().toISOString(),
        sets: entrySets.map((s) => ({
          reps: parseInt(s.reps, 10),
          weight_kg: parseFloat(s.weight_kg),
          rpe: s.rpe ? parseFloat(s.rpe) : undefined,
        })),
        notes: entryNotes || undefined,
      });
      setEntrySets([{ reps: '', weight_kg: '', rpe: '' }]);
      setEntryNotes('');
      setShowEntryForm(false);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to log entry');
    }
  };

  const updateEntrySet = (idx: number, field: keyof SetRow, val: string) => {
    setEntrySets((prev) => prev.map((s, i) => i === idx ? { ...s, [field]: val } : s));
  };

  const handleGetAdvice = async () => {
    setError('');
    try { await getAdvice(auth.userId!); }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed to get advice'); }
  };

  if (!initialized) return <PageLoader />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--text-primary)]">Goal</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">Track progress toward your target lift</p>
        </div>
        {goal && (
          <Button variant="ghost" size="sm" onClick={() => setShowGoalForm(!showGoalForm)}>
            Change Goal
          </Button>
        )}
      </div>

      {/* No goal state */}
      {!goal && !showGoalForm && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-14 h-14 rounded-2xl bg-[var(--bg-elevated)] border border-[var(--border)] flex items-center justify-center mb-4">
            <Target size={24} className="text-[var(--text-muted)]" />
          </div>
          <h3 className="text-base font-semibold text-[var(--text-primary)] mb-1">No active goal</h3>
          <p className="text-sm text-[var(--text-muted)] mb-4">Pick an exercise and set a target to start tracking</p>
          <Button variant="primary" onClick={() => setShowGoalForm(true)}>
            <Plus size={15} /> Set a Goal
          </Button>
        </div>
      )}

      {/* Goal Form */}
      {showGoalForm && (
        <Card>
          <h2 className="text-base font-semibold text-[var(--text-primary)] mb-4">
            {goal ? 'Change Goal' : 'Set a Goal'}
          </h2>
          <form onSubmit={handleSetGoal} className="space-y-4">
            {/* Exercise Picker */}
            <div className="flex flex-col gap-1.5" ref={dropRef}>
              <label className="text-sm font-medium text-[var(--text-secondary)]">Exercise</label>
              <div className="relative">
                <input
                  value={goalQuery}
                  onChange={(e) => handleGoalSearch(e.target.value)}
                  onFocus={() => goalQuery.length >= 2 && setGoalDropOpen(true)}
                  placeholder="Search exercise…"
                  className="w-full px-3 py-2.5 pr-8 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent transition-all"
                />
                <ChevronDown size={14} className="absolute right-3 top-3.5 text-[var(--text-muted)]" />
                {goalDropOpen && goalResults.length > 0 && (
                  <div className="absolute z-50 mt-1 w-full rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] shadow-xl max-h-48 overflow-y-auto">
                    {goalResults.map((ex) => (
                      <button
                        key={ex.id}
                        type="button"
                        onClick={() => {
                          setSelectedExercise(ex);
                          setGoalQuery(ex.name);
                          setGoalDropOpen(false);
                        }}
                        className="flex items-center justify-between w-full px-3 py-2.5 text-sm text-left hover:bg-[var(--bg-elevated)] transition-colors"
                      >
                        <span className="text-[var(--text-primary)] capitalize">{ex.name}</span>
                        <span className="text-xs text-[var(--text-muted)]">{ex.bodyPart}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <Input
                label="Target weight (kg)"
                type="number"
                min={0}
                step={0.5}
                value={targetWeight}
                onChange={(e) => setTargetWeight(e.target.value)}
                placeholder="100"
              />
              <Input
                label="Target reps"
                type="number"
                min={1}
                value={targetReps}
                onChange={(e) => setTargetReps(e.target.value)}
                placeholder="1"
              />
            </div>
            <Input
              label="Deadline (optional)"
              type="date"
              value={deadline}
              onChange={(e) => setDeadline(e.target.value)}
              min={new Date().toISOString().split('T')[0]}
            />
            <Input
              label="Notes (optional)"
              value={goalNotes}
              onChange={(e) => setGoalNotes(e.target.value)}
              placeholder="e.g. preparing for powerlifting meet"
            />

            {error && (
              <p className="text-sm text-[var(--danger)] bg-[var(--danger-muted)] px-3 py-2 rounded-lg">
                {error}
              </p>
            )}

            <div className="flex gap-3">
              <Button type="submit" variant="primary" loading={loading}>
                Save Goal
              </Button>
              <Button type="button" variant="ghost" onClick={() => { setShowGoalForm(false); setError(''); }}>
                Cancel
              </Button>
            </div>
          </form>
        </Card>
      )}

      {/* Active goal */}
      {goal && !showGoalForm && (
        <>
          <GoalCard goal={goal} />

          {/* Log Entry Section */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider">Log Entry</h2>
              <Button variant="secondary" size="sm" onClick={() => setShowEntryForm(!showEntryForm)}>
                <Plus size={14} /> Add entry
              </Button>
            </div>

            {showEntryForm && (
              <Card className="mb-4">
                <form onSubmit={handleLogEntry} className="space-y-4">
                  <div className="space-y-2">
                    <div className="grid grid-cols-[auto_1fr_1fr_1fr_auto] gap-2 text-xs text-[var(--text-muted)] px-1">
                      <span className="w-8">Set</span><span>Reps</span><span>Weight (kg)</span><span>RPE (opt.)</span><span className="w-6" />
                    </div>
                    {entrySets.map((set, idx) => (
                      <div key={idx} className="grid grid-cols-[auto_1fr_1fr_1fr_auto] gap-2 items-center">
                        <span className="w-8 text-center text-xs text-[var(--text-muted)] font-mono">{idx + 1}</span>
                        <input type="number" min={1} value={set.reps} onChange={(e) => updateEntrySet(idx, 'reps', e.target.value)} placeholder="5"
                          className="w-full px-2.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent" />
                        <input type="number" min={0} step={0.5} value={set.weight_kg} onChange={(e) => updateEntrySet(idx, 'weight_kg', e.target.value)} placeholder="100"
                          className="w-full px-2.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent" />
                        <input type="number" min={1} max={10} step={0.5} value={set.rpe} onChange={(e) => updateEntrySet(idx, 'rpe', e.target.value)} placeholder="8"
                          className="w-full px-2.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent" />
                        <button type="button" onClick={() => entrySets.length > 1 && setEntrySets(prev => prev.filter((_, i) => i !== idx))}
                          className="w-6 h-6 flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--danger)] disabled:opacity-30 cursor-pointer" disabled={entrySets.length <= 1}>
                          <Trash2 size={13} />
                        </button>
                      </div>
                    ))}
                    <Button type="button" variant="ghost" size="sm" onClick={() => setEntrySets(prev => [...prev, { reps: '', weight_kg: '', rpe: '' }])}>
                      <Plus size={13} /> Add set
                    </Button>
                  </div>
                  <Input label="Notes (optional)" value={entryNotes} onChange={(e) => setEntryNotes(e.target.value)} placeholder="How did it feel?" />
                  {error && <p className="text-sm text-[var(--danger)] bg-[var(--danger-muted)] px-3 py-2 rounded-lg">{error}</p>}
                  <div className="flex gap-3">
                    <Button type="submit" variant="primary" loading={loading}>Save Entry</Button>
                    <Button type="button" variant="ghost" onClick={() => setShowEntryForm(false)}>Cancel</Button>
                  </div>
                </form>
              </Card>
            )}
          </div>

          {/* Entry History */}
          {entries.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">Entry History</h2>
              <Card>
                {entries.map((entry) => (
                  <GoalEntryRow key={entry.entry_id} entry={entry} />
                ))}
              </Card>
            </div>
          )}

          {/* AI Advice */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider">AI Advice</h2>
              <Button variant="primary" size="sm" loading={loading} onClick={handleGetAdvice}>
                <Sparkles size={14} /> Analyse Progress
              </Button>
            </div>

            {advice && (
              <div className="space-y-4">
                <div className="rounded-xl border border-[var(--accent-border)] bg-[var(--accent-muted)] p-5">
                  <p className="text-sm font-medium text-[var(--accent-hover)] mb-2">Analysis</p>
                  <p className="text-sm text-[var(--text-primary)] leading-relaxed">{advice.advice}</p>
                </div>
                <div className="rounded-xl border border-[var(--success)]/30 bg-[var(--success-muted)] p-5">
                  <p className="text-sm font-medium text-[var(--success)] mb-2">Next Session</p>
                  <p className="text-sm text-[var(--text-primary)] leading-relaxed">{advice.next_session_suggestion}</p>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
