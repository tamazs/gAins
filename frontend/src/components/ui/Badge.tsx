import React from 'react';

type Color = 'default' | 'accent' | 'success' | 'danger' | 'warning';

interface BadgeProps {
  children: React.ReactNode;
  color?: Color;
  className?: string;
}

const colorStyles: Record<Color, string> = {
  default: 'bg-[var(--bg-elevated)] text-[var(--text-secondary)] border-[var(--border)]',
  accent: 'bg-[var(--accent-muted)] text-[var(--accent-hover)] border-[var(--accent-border)]',
  success: 'bg-[var(--success-muted)] text-[var(--success)] border-[var(--success)]/30',
  danger: 'bg-[var(--danger-muted)] text-[var(--danger)] border-[var(--danger)]/30',
  warning: 'bg-[var(--warning-muted)] text-[var(--warning)] border-[var(--warning)]/30',
};

export function Badge({ children, color = 'default', className = '' }: BadgeProps) {
  return (
    <span
      className={[
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border',
        colorStyles[color],
        className,
      ].join(' ')}
    >
      {children}
    </span>
  );
}
