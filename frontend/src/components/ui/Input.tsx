import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
}

export function Input({ label, error, hint, className = '', id, ...props }: InputProps) {
  const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');
  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label htmlFor={inputId} className="text-sm font-medium text-[var(--text-secondary)]">
          {label}
        </label>
      )}
      <input
        id={inputId}
        {...props}
        className={[
          'w-full px-3 py-2.5 rounded-lg border bg-[var(--bg-elevated)] text-[var(--text-primary)]',
          'placeholder:text-[var(--text-muted)] text-sm',
          'focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent',
          'transition-all duration-150',
          error
            ? 'border-[var(--danger)] focus:ring-[var(--danger)]'
            : 'border-[var(--border)] hover:border-[var(--border-hover)]',
          className,
        ].join(' ')}
      />
      {error && <p className="text-xs text-[var(--danger)]">{error}</p>}
      {hint && !error && <p className="text-xs text-[var(--text-muted)]">{hint}</p>}
    </div>
  );
}

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
}

export function Textarea({ label, error, className = '', id, ...props }: TextareaProps) {
  const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');
  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label htmlFor={inputId} className="text-sm font-medium text-[var(--text-secondary)]">
          {label}
        </label>
      )}
      <textarea
        id={inputId}
        {...props}
        className={[
          'w-full px-3 py-2.5 rounded-lg border bg-[var(--bg-elevated)] text-[var(--text-primary)]',
          'placeholder:text-[var(--text-muted)] text-sm resize-none',
          'focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent',
          'transition-all duration-150',
          error
            ? 'border-[var(--danger)]'
            : 'border-[var(--border)] hover:border-[var(--border-hover)]',
          className,
        ].join(' ')}
      />
      {error && <p className="text-xs text-[var(--danger)]">{error}</p>}
    </div>
  );
}
