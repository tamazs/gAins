import { useState, useCallback } from 'react';
import { setGoal, getGoal, logGoalEntry, analyseGoal, getGoalEntries } from '../api/goals';
import type {
  GoalRequest, GoalResponse,
  GoalEntryRequest, GoalEntryResponse,
  GoalAdviceResponse,
} from '../types/api';

export function useGoal() {
  const [goal, setGoalState] = useState<GoalResponse | null>(null);
  const [entries, setEntries] = useState<GoalEntryResponse[]>([]);
  const [advice, setAdvice] = useState<GoalAdviceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGoal = useCallback(async (userId: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getGoal(userId);
      setGoalState(data);
    } catch (e: unknown) {
      if ((e as { response?: { status: number } })?.response?.status === 404) {
        setGoalState(null);
      } else {
        setError(e instanceof Error ? e.message : 'Failed to load goal');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchEntries = useCallback(async (userId: string) => {
    setError(null);
    try {
      const data = await getGoalEntries(userId);
      setEntries(data);
    } catch (e: unknown) {
      // 404 means no goal yet — not an error worth surfacing
      if ((e as { response?: { status: number } })?.response?.status !== 404) {
        setError(e instanceof Error ? e.message : 'Failed to load entries');
      }
    }
  }, []);

  const createGoal = useCallback(async (req: GoalRequest) => {
    setLoading(true);
    setError(null);
    try {
      const data = await setGoal(req);
      setGoalState(data);
      setEntries([]); // clear entries — it's a new goal
      return data;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to set goal';
      setError(msg);
      throw new Error(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  const addEntry = useCallback(async (req: GoalEntryRequest) => {
    setLoading(true);
    setError(null);
    try {
      const data = await logGoalEntry(req);
      // Prepend to the list — single source of truth, no extra setState in the page
      setEntries((prev) => [data, ...prev]);
      return data;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to log entry';
      setError(msg);
      throw new Error(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  const getAdvice = useCallback(async (userId: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyseGoal(userId);
      setAdvice(data);
      return data;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to get advice';
      setError(msg);
      throw new Error(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    goal,
    entries,
    advice,
    loading,
    error,
    fetchGoal,
    fetchEntries,
    createGoal,
    addEntry,
    getAdvice,
  };
}
