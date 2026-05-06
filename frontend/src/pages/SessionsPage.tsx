import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, ClipboardList } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { useSessions } from '../hooks/useSessions';
import { SessionCard } from '../components/sessions/SessionCard';
import { Button } from '../components/ui/Button';
import { PageLoader } from '../components/ui/Spinner';

export function SessionsPage() {
  const { auth } = useAuth();
  const { sessions, loading, fetchSessions } = useSessions();
  const navigate = useNavigate();

  useEffect(() => {
    if (auth.userId) fetchSessions(auth.userId);
  }, [auth.userId]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--text-primary)]">Sessions</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            {sessions.length} session{sessions.length !== 1 ? 's' : ''} logged
          </p>
        </div>
        <Button variant="primary" onClick={() => navigate('/sessions/new')}>
          <Plus size={15} /> Log Session
        </Button>
      </div>

      {/* List */}
      {loading ? (
        <PageLoader />
      ) : sessions.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-14 h-14 rounded-2xl bg-[var(--bg-elevated)] border border-[var(--border)] flex items-center justify-center mb-4">
            <ClipboardList size={24} className="text-[var(--text-muted)]" />
          </div>
          <h3 className="text-base font-semibold text-[var(--text-primary)] mb-1">No sessions yet</h3>
          <p className="text-sm text-[var(--text-muted)] mb-4">Log your first workout to get AI-powered feedback</p>
          <Button variant="primary" onClick={() => navigate('/sessions/new')}>
            <Plus size={15} /> Log your first session
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          {sessions.map((session, i) => (
            <SessionCard key={i} session={session} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
