import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Dumbbell, LayoutDashboard, ClipboardList, Target, LogOut } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

const NAV_ITEMS = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/sessions', label: 'Sessions', icon: ClipboardList },
  { to: '/goals', label: 'Goals', icon: Target },
];

export function Navbar() {
  const { auth, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <aside className="fixed left-0 top-0 h-full w-56 flex flex-col bg-[var(--bg-surface)] border-r border-[var(--border)] z-50">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 py-5 border-b border-[var(--border)]">
        <div className="w-8 h-8 rounded-lg bg-[var(--accent)] flex items-center justify-center flex-shrink-0">
          <Dumbbell size={16} className="text-white" />
        </div>
        <span className="text-base font-bold text-[var(--text-primary)] tracking-tight">gAin</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
          const active = location.pathname.startsWith(to);
          return (
            <Link
              key={to}
              to={to}
              className={[
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150',
                active
                  ? 'bg-[var(--accent-muted)] text-[var(--accent-hover)] border border-[var(--accent-border)]'
                  : 'text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]',
              ].join(' ')}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* User */}
      <div className="px-3 py-4 border-t border-[var(--border)]">
        <div className="flex items-center gap-2 px-3 py-2 mb-2">
          <div className="w-7 h-7 rounded-full bg-[var(--accent-muted)] border border-[var(--accent-border)] flex items-center justify-center flex-shrink-0">
            <span className="text-xs font-bold text-[var(--accent-hover)]">
              {auth.username?.charAt(0).toUpperCase() ?? '?'}
            </span>
          </div>
          <span className="text-sm text-[var(--text-secondary)] truncate">{auth.username}</span>
        </div>
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm text-[var(--text-muted)] hover:text-[var(--danger)] hover:bg-[var(--danger-muted)] transition-all duration-150 cursor-pointer"
        >
          <LogOut size={15} />
          Sign out
        </button>
      </div>
    </aside>
  );
}
