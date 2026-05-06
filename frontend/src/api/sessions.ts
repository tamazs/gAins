import client from './client';
import type { WorkoutSessionRequest, WorkoutAdviceResponse, SessionDocument } from '../types/api';

export async function createSession(session: WorkoutSessionRequest): Promise<WorkoutAdviceResponse> {
  const { data } = await client.post<WorkoutAdviceResponse>('/sessions', session);
  return data;
}

export async function getSessions(userId: string, limit = 20): Promise<SessionDocument[]> {
  const { data } = await client.get<SessionDocument[]>(`/sessions/${userId}`, {
    params: { limit },
  });
  return data;
}
