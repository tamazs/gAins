import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, BookOpen, Sparkles } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { useSessions } from '../hooks/useSessions';
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

      {/* AI Analysis */}
      {session.analysis && (
        <div>
          <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3 flex items-center gap-2">
            <Sparkles size={14} /> AI Analysis
          </h2>

          <div className="space-y-4">
            <div className="rounded-xl border border-[var(--accent-border)] bg-[var(--accent-muted)] p-5">
              <p className="text-sm font-medium text-[var(--accent-hover)] mb-2">Overall Summary</p>
              <p className="text-sm text-[var(--text-primary)] leading-relaxed">{session.analysis.overall_summary}</p>
            </div>

            {session.analysis.recovery_flag && (
              <div className="rounded-xl border border-[var(--danger)]/30 bg-[var(--danger-muted)] p-4">
                <p className="text-sm font-medium text-[var(--danger)]">⚠️ Recovery Warning</p>
                <p className="text-sm text-[var(--text-secondary)] mt-1">Signs of potential overtraining detected. Consider additional rest.</p>
              </div>
            )}

            {session.analysis.exercise_advice.map((ea) => (
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

            {session.analysis.sources_used.length > 0 && (
              <div className="flex items-start gap-2 text-xs text-[var(--text-muted)]">
                <BookOpen size={13} className="mt-0.5 shrink-0" />
                <span>Sources: {session.analysis.sources_used.join(', ')}</span>
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
