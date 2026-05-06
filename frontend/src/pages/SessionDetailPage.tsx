import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, AlertTriangle, BookOpen, TrendingUp } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { useSessions } from '../hooks/useSessions';
import { ProgressChart } from '../components/sessions/ProgressChart';
import { Badge } from '../components/ui/Badge';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { PageLoader } from '../components/ui/Spinner';

export function SessionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { auth } = useAuth();
  const { sessions, fetchSessions, loading } = useSessions();
  const navigate = useNavigate();
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    if (auth.userId) {
      fetchSessions(auth.userId).finally(() => setInitialized(true));
    }
  }, [auth.userId]);

  if (!initialized || loading) return <PageLoader />;

  const idx = parseInt(id ?? '0', 10);
  const session = sessions[idx];

  if (!session) {
    return (
      <div className="text-center py-16">
        <p className="text-[var(--text-muted)]">Session not found.</p>
        <Button variant="ghost" className="mt-4" onClick={() => navigate('/sessions')}>Back to Sessions</Button>
      </div>
    );
  }

  const date = new Date(session.date).toLocaleDateString('en-GB', {
    weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
  });

  // The advice comes embedded in the session document from the backend
  // (not stored as a separate field — we just show session data here)

  return (
    <div className="space-y-6">
      {/* Back */}
      <button
        onClick={() => navigate('/sessions')}
        className="flex items-center gap-2 text-sm text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
      >
        <ArrowLeft size={15} /> Back to sessions
      </button>

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Session Detail</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">{date}</p>
        {session.notes && (
          <p className="text-sm text-[var(--text-secondary)] mt-2 italic">"{session.notes}"</p>
        )}
      </div>

      {/* Exercises */}
      <div>
        <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">Exercises</h2>
        <div className="space-y-4">
          {session.exercises.map((ex) => (
            <Card key={ex.name}>
              <div className="flex items-center gap-3 mb-3">
                <h3 className="text-base font-semibold text-[var(--text-primary)]">{ex.name}</h3>
                <Badge color="default">{ex.muscle_group}</Badge>
              </div>
              <div className="space-y-1.5">
                {ex.sets.map((set, si) => (
                  <div key={si} className="flex items-center gap-4 text-sm">
                    <span className="text-[var(--text-muted)] w-12">Set {si + 1}</span>
                    <span className="text-[var(--text-primary)]">{set.reps} reps @ {set.weight_kg} kg</span>
                    {set.rpe != null && (
                      <Badge color={set.rpe >= 9 ? 'danger' : set.rpe >= 7 ? 'warning' : 'success'}>
                        RPE {set.rpe}
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Progression Charts */}
      {sessions.length >= 2 && (
        <div>
          <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3 flex items-center gap-2">
            <TrendingUp size={14} /> Progression
          </h2>
          <div className="space-y-4">
            {session.exercises.map((ex) => (
              <Card key={ex.name}>
                <p className="text-sm font-medium text-[var(--text-secondary)] mb-3">{ex.name}</p>
                <ProgressChart sessions={sessions} exerciseName={ex.name} />
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
