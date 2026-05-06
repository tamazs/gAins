import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { Navbar } from './Navbar';

function getLocalAuth(): boolean {
  try {
    const raw = localStorage.getItem('auth');
    if (!raw) return false;
    return !!JSON.parse(raw)?.token;
  } catch {
    return false;
  }
}

export function PrivateRoute() {
  const { isAuthenticated } = useAuth();
  // Jotai's atomWithStorage may not have synced yet on first render;
  // fall back to a direct localStorage check so the route doesn't flash-redirect.
  const localAuth = getLocalAuth();

  if (!isAuthenticated && !localAuth) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="flex min-h-screen bg-[var(--bg-base)]">
      <Navbar />
      <main className="flex-1 ml-56 min-h-screen">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
