import { useState, useCallback } from 'react';
import { useAtom } from 'jotai';
import { exerciseCacheAtom, exercisesLoadedAtom } from '../atoms/exercisesAtom';
import { getExercises, searchExercisesByName } from '../api/exercises';
import type { ExerciseDbEntry } from '../types/exercise';

export function useExercises() {
  const [cache, setCache] = useAtom(exerciseCacheAtom);
  const [loaded, setLoaded] = useAtom(exercisesLoadedAtom);
  const [loading, setLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<ExerciseDbEntry[]>([]);

  const loadInitial = useCallback(async () => {
    if (loaded) return;
    setLoading(true);
    try {
      // Load first 100 exercises to populate cache
      const data = await getExercises(100, 0);
      setCache(data);
      setLoaded(true);
    } catch {
      // silently fail – user can still search
    } finally {
      setLoading(false);
    }
  }, [loaded, setCache, setLoaded]);

  const search = useCallback(
    async (query: string): Promise<ExerciseDbEntry[]> => {
      if (!query.trim()) {
        setSearchResults([]);
        return [];
      }
      // First try local cache
      const local = cache.filter((e) =>
        e.name.toLowerCase().includes(query.toLowerCase())
      );
      if (local.length > 0) {
        setSearchResults(local.slice(0, 20));
        return local.slice(0, 20);
      }
      // Fall back to API search
      setLoading(true);
      try {
        const results = await searchExercisesByName(query, 20);
        setSearchResults(results);
        return results;
      } catch {
        return [];
      } finally {
        setLoading(false);
      }
    },
    [cache]
  );

  return { exercises: cache, loading, loadInitial, search, searchResults };
}
