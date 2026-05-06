import { useState, useRef, useEffect } from 'react';
import { Plus, Trash2, ChevronDown } from 'lucide-react';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Badge } from '../ui/Badge';
import { useExercises } from '../../hooks/useExercises';
import { toMuscleGroup, MUSCLE_GROUP_LABELS } from '../../utils/muscleGroupMap';
import type { ExerciseDbEntry } from '../../types/exercise';

export interface SetRow {
  reps: string;
  weight_kg: string;
  rpe: string;
}

export interface ExerciseRow {
  id: string;
  exerciseName: string;
  muscleGroup: string;
  sets: SetRow[];
}

interface Props {
  row: ExerciseRow;
  onChange: (updated: ExerciseRow) => void;
  onRemove: () => void;
  showRemove: boolean;
}

export function ExerciseFormRow({ row, onChange, onRemove, showRemove }: Props) {
  const [query, setQuery] = useState(row.exerciseName);
  const [open, setOpen] = useState(false);
  const [results, setResults] = useState<ExerciseDbEntry[]>([]);
  const { search, exercises } = useExercises();
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleSearch = async (q: string) => {
    setQuery(q);
    setOpen(true);
    if (q.length < 2) { setResults([]); return; }
    const found = await search(q);
    setResults(found.slice(0, 10));
  };

  const selectExercise = (ex: ExerciseDbEntry) => {
    const mg = toMuscleGroup(ex.bodyPart, ex.target);
    onChange({ ...row, exerciseName: ex.name, muscleGroup: mg });
    setQuery(ex.name);
    setOpen(false);
    setResults([]);
  };

  // Also search in initial cache when query changes
  useEffect(() => {
    if (query.length >= 2 && exercises.length > 0) {
      const local = exercises.filter(e => e.name.toLowerCase().includes(query.toLowerCase())).slice(0, 10);
      setResults(local);
    }
  }, [query, exercises]);

  const updateSet = (idx: number, field: keyof SetRow, val: string) => {
    const sets = row.sets.map((s, i) => i === idx ? { ...s, [field]: val } : s);
    onChange({ ...row, sets });
  };

  const addSet = () => {
    onChange({ ...row, sets: [...row.sets, { reps: '', weight_kg: '', rpe: '' }] });
  };

  const removeSet = (idx: number) => {
    if (row.sets.length <= 1) return;
    onChange({ ...row, sets: row.sets.filter((_, i) => i !== idx) });
  };

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-4 space-y-4">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="flex-1 relative" ref={dropdownRef}>
          <div className="relative">
            <input
              value={query}
              onChange={(e) => handleSearch(e.target.value)}
              onFocus={() => query.length >= 2 && setOpen(true)}
              placeholder="Search exercise (e.g. bench press)…"
              className="w-full px-3 py-2.5 pr-8 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent transition-all"
            />
            <ChevronDown size={14} className="absolute right-3 top-3.5 text-[var(--text-muted)]" />
          </div>
          {open && results.length > 0 && (
            <div className="absolute z-50 mt-1 w-full rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] shadow-xl max-h-56 overflow-y-auto">
              {results.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  onClick={() => selectExercise(ex)}
                  className="flex items-center justify-between w-full px-3 py-2.5 text-sm text-left hover:bg-[var(--bg-elevated)] transition-colors"
                >
                  <span className="text-[var(--text-primary)] capitalize">{ex.name}</span>
                  <span className="text-xs text-[var(--text-muted)] ml-2 shrink-0">{ex.bodyPart}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        {row.muscleGroup && (
          <Badge color="accent" className="mt-2 shrink-0">
            {MUSCLE_GROUP_LABELS[row.muscleGroup] ?? row.muscleGroup}
          </Badge>
        )}
        {showRemove && (
          <button
            type="button"
            onClick={onRemove}
            className="mt-2 p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--danger)] hover:bg-[var(--danger-muted)] transition-all cursor-pointer shrink-0"
          >
            <Trash2 size={15} />
          </button>
        )}
      </div>

      {/* Sets */}
      <div className="space-y-2">
        <div className="grid grid-cols-[auto_1fr_1fr_1fr_auto] gap-2 items-center text-xs text-[var(--text-muted)] px-1">
          <span className="w-8">Set</span>
          <span>Reps</span>
          <span>Weight (kg)</span>
          <span>RPE (opt.)</span>
          <span className="w-6" />
        </div>
        {row.sets.map((set, idx) => (
          <div key={idx} className="grid grid-cols-[auto_1fr_1fr_1fr_auto] gap-2 items-center">
            <span className="w-8 text-center text-xs text-[var(--text-muted)] font-mono">{idx + 1}</span>
            <input
              type="number"
              min={1}
              value={set.reps}
              onChange={(e) => updateSet(idx, 'reps', e.target.value)}
              placeholder="10"
              className="w-full px-2.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent transition-all"
            />
            <input
              type="number"
              min={0}
              step={0.5}
              value={set.weight_kg}
              onChange={(e) => updateSet(idx, 'weight_kg', e.target.value)}
              placeholder="60"
              className="w-full px-2.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent transition-all"
            />
            <input
              type="number"
              min={1}
              max={10}
              step={0.5}
              value={set.rpe}
              onChange={(e) => updateSet(idx, 'rpe', e.target.value)}
              placeholder="8"
              className="w-full px-2.5 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent transition-all"
            />
            <button
              type="button"
              onClick={() => removeSet(idx)}
              disabled={row.sets.length <= 1}
              className="w-6 h-6 flex items-center justify-center rounded text-[var(--text-muted)] hover:text-[var(--danger)] disabled:opacity-30 transition-colors cursor-pointer"
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
        <Button type="button" variant="ghost" size="sm" onClick={addSet} className="text-xs">
          <Plus size={13} /> Add set
        </Button>
      </div>
    </div>
  );
}
