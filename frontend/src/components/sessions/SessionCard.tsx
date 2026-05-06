import { useNavigate } from 'react-router-dom';
import { Calendar, ChevronRight, Dumbbell } from 'lucide-react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import type { SessionDocument } from '../../types/api';

interface Props {
  session: SessionDocument;
  index: number;
}

export function SessionCard({ session, index }: Props) {
  const navigate = useNavigate();
  const date = new Date(session.date).toLocaleDateString('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric',
  });

  return (
    <Card
      hover
      onClick={() => navigate(`/sessions/${index}`)}
      className="group"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-2 flex-1 min-w-0">
          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
            <Calendar size={12} />
            <span>{date}</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {session.exercises?.slice(0, 4).map((ex) => (
              <Badge key={ex.name} color="default">
                {ex.name}
              </Badge>
            ))}
            {(session.exercises?.length ?? 0) > 4 && (
              <Badge color="default">+{session.exercises.length - 4} more</Badge>
            )}
          </div>
          {session.notes && (
            <p className="text-xs text-[var(--text-muted)] truncate">{session.notes}</p>
          )}
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
            <Dumbbell size={13} />
            <span>{session.exercises?.length ?? 0} exercises</span>
          </div>
          <ChevronRight size={15} className="text-[var(--text-muted)] group-hover:text-[var(--accent)] transition-colors" />
        </div>
      </div>
    </Card>
  );
}
