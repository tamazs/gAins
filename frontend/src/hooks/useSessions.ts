import { useState, useCallback } from 'react';
import { createSession, getSessions } from '../api/sessions';
import type { WorkoutSessionRequest, WorkoutAdviceResponse, SessionDocument } from '../types/api';

export function useSessions() {
  const [sessions, setSessions] = useState<SessionDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSessions = useCallback(async (userId: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSessions(userId);
      setSessions(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load sessions');
    } finally {
      setLoading(false);
    }
  }, []);

  const submitSession = useCallback(
    async (session: WorkoutSessionRequest): Promise<WorkoutAdviceResponse> => {
      setLoading(true);
      setError(null);
      try {
        const result = await createSession(session);
        return result;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Failed to submit session';
        setError(msg);
        throw new Error(msg);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { sessions, loading, error, fetchSessions, submitSession };
}
