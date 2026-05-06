import { atom } from 'jotai';
import type { ExerciseDbEntry } from '../types/exercise';

export const exerciseCacheAtom = atom<ExerciseDbEntry[]>([]);
export const exercisesLoadedAtom = atom(false);
