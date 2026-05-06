/**
 * Maps ExerciseDB bodyPart + target → backend muscle_group enum
 * Backend valid values: chest, back, shoulders, biceps, triceps,
 *   legs, quads, hamstrings, glutes, calves, core
 */
export function toMuscleGroup(bodyPart: string, target: string): string {
  const bp = bodyPart.toLowerCase();
  const tgt = target.toLowerCase();

  if (bp === 'chest') return 'chest';
  if (bp === 'back') return 'back';
  if (bp === 'shoulders') return 'shoulders';
  if (bp === 'waist') return 'core';
  if (bp === 'lower arms') return 'core';
  if (bp === 'neck') return 'back';
  if (bp === 'cardio') return 'legs';

  if (bp === 'upper arms') {
    if (tgt.includes('tricep')) return 'triceps';
    return 'biceps';
  }

  if (bp === 'upper legs') {
    if (tgt.includes('hamstring')) return 'hamstrings';
    if (tgt.includes('glute')) return 'glutes';
    return 'quads';
  }

  if (bp === 'lower legs') {
    if (tgt.includes('glute')) return 'glutes';
    return 'calves';
  }

  return 'legs'; // fallback
}

export const MUSCLE_GROUP_LABELS: Record<string, string> = {
  chest: 'Chest',
  back: 'Back',
  shoulders: 'Shoulders',
  biceps: 'Biceps',
  triceps: 'Triceps',
  legs: 'Legs',
  quads: 'Quads',
  hamstrings: 'Hamstrings',
  glutes: 'Glutes',
  calves: 'Calves',
  core: 'Core',
};
