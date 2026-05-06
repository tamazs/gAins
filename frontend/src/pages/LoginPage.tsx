import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Dumbbell } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/dashboard');
    } catch {
      setError('Invalid email or password.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--bg-base)] px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex items-center justify-center gap-2.5 mb-8">
          <div className="w-10 h-10 rounded-xl bg-[var(--accent)] flex items-center justify-center">
            <Dumbbell size={20} className="text-white" />
          </div>
          <span className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">gAin</span>
        </div>

        {/* Card */}
        <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-surface)] p-8 shadow-2xl">
          <h1 className="text-xl font-semibold text-[var(--text-primary)] mb-1">Welcome back</h1>
          <p className="text-sm text-[var(--text-muted)] mb-6">Sign in to your account</p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              label="Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              autoComplete="email"
            />
            <Input
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              autoComplete="current-password"
            />

            {error && (
              <p className="text-sm text-[var(--danger)] bg-[var(--danger-muted)] px-3 py-2 rounded-lg border border-[var(--danger)]/20">
                {error}
              </p>
            )}

            <Button type="submit" variant="primary" size="lg" loading={loading} className="w-full mt-2">
              Sign in
            </Button>
          </form>

          <p className="text-sm text-[var(--text-muted)] text-center mt-6">
            No account?{' '}
            <Link to="/register" className="text-[var(--accent-hover)] hover:underline">
              Create one
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
