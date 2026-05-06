import React from 'react';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  children: React.ReactNode;
}

const variantStyles: Record<Variant, string> = {
  primary: 'bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white border-transparent',
  secondary: 'bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] text-[var(--text-primary)] border-[var(--border-hover)]',
  ghost: 'bg-transparent hover:bg-[var(--bg-elevated)] text-[var(--text-secondary)] border-transparent',
  danger: 'bg-[var(--danger-muted)] hover:bg-red-900/30 text-[var(--danger)] border-[var(--danger)]/30',
};

const sizeStyles: Record<Size, string> = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
};

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  children,
  className = '',
  ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      disabled={disabled || loading}
      className={[
        'inline-flex items-center justify-center gap-2 rounded-lg border font-medium transition-all duration-150',
        'disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer',
        variantStyles[variant],
        sizeStyles[size],
        className,
      ].join(' ')}
    >
      {loading && (
        <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
      )}
      {children}
    </button>
  );
}
