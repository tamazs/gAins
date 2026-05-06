import { Target, Calendar, FileText } from 'lucide-react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import type { GoalResponse } from '../../types/api';
import { MUSCLE_GROUP_LABELS } from '../../utils/muscleGroupMap';

interface Props {
  goal: GoalResponse;
}

export function GoalCard({ goal }: Props) {
  const deadline = goal.deadline
    ? new Date(goal.deadline).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
    : null;

  return (
    <Card>
      <div className="flex items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-[var(--accent-muted)] border border-[var(--accent-border)] flex items-center justify-center shrink-0">
          <Target size={18} className="text-[var(--accent-hover)]" />
        </div>
        <div className="flex-1 space-y-3">
          <div className="flex items-center gap-3 flex-wrap">
            <h3 className="text-base font-semibold text-[var(--text-primary)]">{goal.exercise_name}</h3>
            <Badge color="accent">{MUSCLE_GROUP_LABELS[goal.muscle_group] ?? goal.muscle_group}</Badge>
          </div>
          <div className="flex items-center gap-6">
            <div>
              <p className="text-xs text-[var(--text-muted)] mb-0.5">Target</p>
              <p className="text-2xl font-bold text-[var(--text-primary)]">
                {goal.target_reps}
                <span className="text-base font-normal text-[var(--text-secondary)] ml-1">
                  rep{goal.target_reps !== 1 ? 's' : ''}
                </span>
                <span className="text-[var(--text-muted)] mx-2">@</span>
                {goal.target_weight_kg}
                <span className="text-base font-normal text-[var(--text-secondary)] ml-1">kg</span>
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-4 text-sm text-[var(--text-muted)]">
            {deadline && (
              <div className="flex items-center gap-1.5">
                <Calendar size={13} />
                <span>Due {deadline}</span>
              </div>
            )}
            {goal.notes && (
              <div className="flex items-center gap-1.5">
                <FileText size={13} />
                <span className="truncate max-w-xs">{goal.notes}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
