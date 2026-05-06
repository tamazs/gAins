import { atomWithStorage } from 'jotai/utils';

interface AuthState {
  token: string | null;
  userId: string | null;
  username: string | null;
}

const DEFAULT: AuthState = { token: null, userId: null, username: null };

export const authAtom = atomWithStorage<AuthState>('auth', DEFAULT);
