import { Calendar } from 'lucide-react';
import type { GoalEntryResponse } from '../../types/api';

interface Props {
  entry: GoalEntryResponse;
}

export function GoalEntryRow({ entry }: Props) {
  const date = new Date(entry.date).toLocaleDateString('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric',
  });

  return (
    <div className="flex items-start gap-4 py-3 border-b border-[var(--border)] last:border-0">
      <div className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] w-28 shrink-0 mt-0.5">
        <Calendar size={12} />
        <span>{date}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {entry.sets.map((set, i) => (
          <span
            key={i}
            className="px-2.5 py-1 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border)] text-xs text-[var(--text-secondary)]"
          >
            {set.reps} × {set.weight_kg}kg
            {set.rpe != null && <span className="text-[var(--text-muted)] ml-1">RPE {set.rpe}</span>}
          </span>
        ))}
      </div>
      {entry.notes && (
        <p className="text-xs text-[var(--text-muted)] ml-auto truncate max-w-xs">{entry.notes}</p>
      )}
    </div>
  );
}
