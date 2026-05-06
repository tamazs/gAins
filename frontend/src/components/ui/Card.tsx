import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  hover?: boolean;
}

export function Card({ children, className = '', onClick, hover }: CardProps) {
  return (
    <div
      onClick={onClick}
      className={[
        'rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] p-5',
        hover ? 'hover:border-[var(--border-hover)] hover:bg-[var(--bg-elevated)] transition-all duration-150 cursor-pointer' : '',
        className,
      ].join(' ')}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <div className={`mb-4 ${className}`}>{children}</div>;
}

export function CardTitle({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <h3 className={`text-base font-semibold text-[var(--text-primary)] ${className}`}>{children}</h3>;
}
