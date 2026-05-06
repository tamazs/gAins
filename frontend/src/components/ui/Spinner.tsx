interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizes = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-10 h-10' };

export function Spinner({ size = 'md', className = '' }: SpinnerProps) {
  return (
    <div
      className={[
        'rounded-full border-2 border-[var(--border)] border-t-[var(--accent)] animate-spin',
        sizes[size],
        className,
      ].join(' ')}
    />
  );
}

export function PageLoader() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[200px] gap-3">
      <Spinner size="lg" />
      <p className="text-sm text-[var(--text-muted)]">Loading…</p>
    </div>
  );
}
