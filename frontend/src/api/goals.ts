import client from './client';
import type {
  GoalRequest, GoalResponse,
  GoalEntryRequest, GoalEntryResponse,
  GoalAdviceResponse,
} from '../types/api';

export async function setGoal(goal: GoalRequest): Promise<GoalResponse> {
  const { data } = await client.post<GoalResponse>('/goals', goal);
  return data;
}

export async function getGoal(userId: string): Promise<GoalResponse> {
  const { data } = await client.get<GoalResponse>(`/goals/${userId}`);
  return data;
}

export async function logGoalEntry(entry: GoalEntryRequest): Promise<GoalEntryResponse> {
  const { data } = await client.post<GoalEntryResponse>('/goals/entries', entry);
  return data;
}

export async function analyseGoal(userId: string): Promise<GoalAdviceResponse> {
  const { data } = await client.post<GoalAdviceResponse>('/goals/analyse', { user_id: userId });
  return data;
}

export async function getGoalEntries(userId: string, limit = 20): Promise<GoalEntryResponse[]> {
  const { data } = await client.get<GoalEntryResponse[]>(`/goals/entries/${userId}`, {
    params: { limit },
  });
  return data;
}

