import { useAtom } from 'jotai';
import { useCallback } from 'react';
import { authAtom } from '../atoms/authAtom';
import { login as apiLogin, register as apiRegister } from '../api/auth';

export function useAuth() {
  const [auth, setAuth] = useAtom(authAtom);

  const login = useCallback(async (email: string, password: string) => {
    const res = await apiLogin(email, password);
    setAuth({ token: res.access_token, userId: res.user_id, username: res.username });
    return res;
  }, [setAuth]);

  const register = useCallback(async (email: string, password: string, username: string) => {
    const res = await apiRegister(email, password, username);
    setAuth({ token: res.access_token, userId: res.user_id, username: res.username });
    return res;
  }, [setAuth]);

  const logout = useCallback(() => {
    setAuth({ token: null, userId: null, username: null });
  }, [setAuth]);

  return {
    auth,
    isAuthenticated: !!auth.token,
    login,
    register,
    logout,
  };
}
