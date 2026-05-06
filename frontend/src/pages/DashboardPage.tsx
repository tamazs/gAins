import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ClipboardList, Target, Plus, TrendingUp, Zap } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { useSessions } from '../hooks/useSessions';
import { useGoal } from '../hooks/useGoal';
import { useExercises } from '../hooks/useExercises';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { PageLoader } from '../components/ui/Spinner';

export function DashboardPage() {
  const { auth } = useAuth();
  const { sessions, fetchSessions, loading: sessionsLoading } = useSessions();
  const { goal, fetchGoal, loading: goalLoading } = useGoal();
  const { loadInitial } = useExercises();
  const navigate = useNavigate();
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    if (auth.userId) {
      Promise.all([
        fetchSessions(auth.userId),
        fetchGoal(auth.userId),
        loadInitial(),
      ]).finally(() => setInitialized(true));
    }
  }, [auth.userId]);

  const loading = !initialized && (sessionsLoading || goalLoading);

  if (loading) return <PageLoader />;

  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <p className="text-sm text-[var(--text-muted)] mb-1">{greeting}</p>
        <h1 className="text-3xl font-bold text-[var(--text-primary)]">
          Hey, {auth.username} 👋
        </h1>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-[var(--accent-muted)] border border-[var(--accent-border)] flex items-center justify-center shrink-0">
            <ClipboardList size={18} className="text-[var(--accent-hover)]" />
          </div>
          <div>
            <p className="text-2xl font-bold text-[var(--text-primary)]">{sessions.length}</p>
            <p className="text-xs text-[var(--text-muted)]">Sessions logged</p>
          </div>
        </Card>

        <Card className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-[var(--success-muted)] border border-[var(--success)]/30 flex items-center justify-center shrink-0">
            <TrendingUp size={18} className="text-[var(--success)]" />
          </div>
          <div>
            <p className="text-2xl font-bold text-[var(--text-primary)]">
              {sessions.reduce((acc, s) => acc + (s.exercises?.length ?? 0), 0)}
            </p>
            <p className="text-xs text-[var(--text-muted)]">Exercises logged</p>
          </div>
        </Card>

        <Card className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-[var(--warning-muted)] border border-[var(--warning)]/30 flex items-center justify-center shrink-0">
            <Target size={18} className="text-[var(--warning)]" />
          </div>
          <div>
            {goal ? (
              <>
                <p className="text-sm font-semibold text-[var(--text-primary)] leading-tight">{goal.exercise_name}</p>
                <p className="text-xs text-[var(--text-muted)]">{goal.target_reps} rep{goal.target_reps !== 1 ? 's' : ''} @ {goal.target_weight_kg}kg</p>
              </>
            ) : (
              <>
                <p className="text-sm font-medium text-[var(--text-muted)]">No active goal</p>
                <p className="text-xs text-[var(--text-muted)]">Set one to track progress</p>
              </>
            )}
          </div>
        </Card>
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">Quick actions</h2>
        <div className="flex flex-wrap gap-3">
          <Button variant="primary" size="md" onClick={() => navigate('/sessions/new')}>
            <Plus size={15} /> Log Session
          </Button>
          <Button variant="secondary" size="md" onClick={() => navigate('/sessions')}>
            <ClipboardList size={15} /> View Sessions
          </Button>
          <Button variant="secondary" size="md" onClick={() => navigate('/goals')}>
            <Target size={15} /> {goal ? 'View Goal' : 'Set a Goal'}
          </Button>
        </div>
      </div>

      {/* Recent Sessions */}
      {sessions.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider">Recent sessions</h2>
            <button
              onClick={() => navigate('/sessions')}
              className="text-xs text-[var(--accent-hover)] hover:underline cursor-pointer"
            >
              View all
            </button>
          </div>
          <div className="space-y-3">
            {sessions.slice(0, 3).map((session, i) => (
              <Card
                key={i}
                hover
                onClick={() => navigate(`/sessions/${i}`)}
              >
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <p className="text-xs text-[var(--text-muted)]">
                      {new Date(session.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {session.exercises?.slice(0, 3).map((ex) => (
                        <Badge key={ex.name} color="default">{ex.name}</Badge>
                      ))}
                    </div>
                  </div>
                  <Zap size={14} className="text-[var(--text-muted)]" />
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
