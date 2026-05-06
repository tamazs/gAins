export interface ExerciseDbEntry {
  id: string;
  name: string;
  bodyPart: string;
  equipment: string;
  target: string;
  gifUrl: string;
  secondaryMuscles: string[];
  instructions: string[];
}
