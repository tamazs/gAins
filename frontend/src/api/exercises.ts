import exerciseClient from './exerciseClient';
import type { ExerciseDbEntry } from '../types/exercise';

export async function getExercises(limit = 20, offset = 0): Promise<ExerciseDbEntry[]> {
  const { data } = await exerciseClient.get<ExerciseDbEntry[]>('/exercises', {
    params: { limit, offset, sortMethod: 'bodyPart', sortOrder: 'ascending' },
  });
  return data;
}

export async function searchExercisesByName(name: string, limit = 20): Promise<ExerciseDbEntry[]> {
  const { data } = await exerciseClient.get<ExerciseDbEntry[]>(`/exercises/name/${encodeURIComponent(name)}`, {
    params: { limit, offset: 0 },
  });
  return data;
}

export async function getBodyParts(): Promise<string[]> {
  const { data } = await exerciseClient.get<string[]>('/exercises/bodyPartList');
  return data;
}

export async function getExercisesByBodyPart(bodyPart: string, limit = 30): Promise<ExerciseDbEntry[]> {
  const { data } = await exerciseClient.get<ExerciseDbEntry[]>(`/exercises/bodyPart/${encodeURIComponent(bodyPart)}`, {
    params: { limit, offset: 0 },
  });
  return data;
}
