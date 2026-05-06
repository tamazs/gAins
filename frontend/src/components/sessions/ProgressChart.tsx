import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import type { SessionDocument } from '../../types/api';

interface Props {
  sessions: SessionDocument[];
  exerciseName: string;
}

interface DataPoint {
  date: string;
  maxWeight: number;
  totalReps: number;
}

export function ProgressChart({ sessions, exerciseName }: Props) {
  // Build data points: for each session that contains this exercise, extract max weight + total reps
  const data: DataPoint[] = sessions
    .filter((s) =>
      s.exercises?.some((e) => e.name.toLowerCase() === exerciseName.toLowerCase())
    )
    .map((s) => {
      const ex = s.exercises.find((e) => e.name.toLowerCase() === exerciseName.toLowerCase())!;
      const maxWeight = Math.max(...ex.sets.map((st) => st.weight_kg));
      const totalReps = ex.sets.reduce((sum, st) => sum + st.reps, 0);
      return {
        date: new Date(s.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }),
        maxWeight,
        totalReps,
      };
    })
    .reverse(); // chronological order

  if (data.length < 2) {
    return (
      <p className="text-xs text-[var(--text-muted)] py-4 text-center">
        Log at least 2 sessions with {exerciseName} to see progression
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <YAxis yAxisId="weight" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <YAxis yAxisId="reps" orientation="right" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            color: 'var(--text-primary)',
            fontSize: '12px',
          }}
        />
        <Legend wrapperStyle={{ fontSize: '11px', color: 'var(--text-secondary)' }} />
        <Line
          yAxisId="weight"
          type="monotone"
          dataKey="maxWeight"
          stroke="var(--accent)"
          strokeWidth={2}
          dot={{ fill: 'var(--accent)', r: 3 }}
          name="Max weight (kg)"
        />
        <Line
          yAxisId="reps"
          type="monotone"
          dataKey="totalReps"
          stroke="var(--success)"
          strokeWidth={2}
          dot={{ fill: 'var(--success)', r: 3 }}
          name="Total reps"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
