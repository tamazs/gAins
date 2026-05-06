import client from './client';
import type { AuthResponse } from '../types/api';

export async function register(email: string, password: string, username: string): Promise<AuthResponse> {
  const { data } = await client.post<AuthResponse>('/auth/register', { email, password, username });
  return data;
}

export async function login(email: string, password: string): Promise<AuthResponse> {
  const { data } = await client.post<AuthResponse>('/auth/login', { email, password });
  return data;
}
