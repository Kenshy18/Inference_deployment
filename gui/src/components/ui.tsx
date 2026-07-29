import { useId, type ReactNode } from "react";
import { ChevronIcon, FolderIcon } from "./Icons";

/** Panel shell: 24px header strip + scrollable body. */
export function Panel({
  title,
  meta,
  actions,
  children,
  className = "",
}: {
  title: string;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel__head">
        <span className="panel__title">{title}</span>
        {meta !== undefined && <span className="panel__meta">{meta}</span>}
        <span className="panel__spacer" />
        {actions}
      </header>
      {children}
    </section>
  );
}

/** One inspector line: fixed-width label, control on the right. */
export function Row({
  label,
  hint,
  title,
  off = false,
  stack = false,
  always = false,
  children,
}: {
  label?: string;
  hint?: string;
  title?: string;
  off?: boolean;
  stack?: boolean;
  always?: boolean;
  children: ReactNode;
}) {
  return (
    <div
      title={title}
      className={[
        "row",
        stack ? "row--stack" : "",
        off ? "is-off" : "",
        always ? "row--always" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {label !== undefined && <span className="row__label">{label}</span>}
      <div className="ctl">{children}</div>
      {hint && <span className="row__hint">{hint}</span>}
    </div>
  );
}

export function SubHead({ children }: { children: ReactNode }) {
  return <div className="subhead">{children}</div>;
}

export function TextInput({
  value,
  onChange,
  placeholder,
  disabled,
  mono = false,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  mono?: boolean;
}) {
  return (
    <input
      className={mono ? "ctl ctl--mono" : "ctl"}
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      spellCheck={false}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function TextArea({
  value,
  onChange,
  placeholder,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
}) {
  return (
    <textarea
      className="ctl ctl--mono ctl--area"
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      spellCheck={false}
      rows={4}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  unit,
  placeholder,
  disabled,
}: {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  placeholder?: string;
  disabled?: boolean;
}) {
  const input = (
    <input
      className="ctl ctl--mono"
      type="number"
      value={value ?? ""}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(event) =>
        onChange(event.target.value === "" ? null : Number(event.target.value))
      }
    />
  );
  if (!unit) {
    return input;
  }
  return (
    <span className="unit">
      {input}
      <span>{unit}</span>
    </span>
  );
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  disabled,
}: {
  value: T;
  options: ReadonlyArray<{ value: T; label: string }>;
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <select
      className="ctl"
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value as T)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function Segment<T extends string>({
  value,
  options,
  onChange,
  disabled,
}: {
  value: T;
  options: ReadonlyArray<{ value: T; label: string; title?: string }>;
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="seg">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          title={option.title ?? option.label}
          disabled={disabled}
          className={value === option.value ? "is-on" : ""}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Check({
  checked,
  onChange,
  label,
  disabled = false,
  title,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  disabled?: boolean;
  title?: string;
}) {
  const id = useId();
  return (
    <label
      className={`check ${disabled ? "is-disabled" : ""}`}
      htmlFor={id}
      title={title ?? label}
    >
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span className="check__box" />
      {label}
    </label>
  );
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step,
  disabled,
  format = (v: number) => v.toFixed(2),
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  format?: (value: number) => string;
}) {
  return (
    <div className="slider">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <output>{format(value)}</output>
    </div>
  );
}

export function PathInput({
  value,
  onChange,
  onBrowse,
  placeholder,
  disabled = false,
  browseDisabled = false,
}: {
  value: string;
  onChange: (value: string) => void;
  onBrowse: () => void;
  placeholder?: string;
  disabled?: boolean;
  browseDisabled?: boolean;
}) {
  return (
    <div className="path">
      <input
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        spellCheck={false}
        title={value}
        onChange={(event) => onChange(event.target.value)}
      />
      <button
        type="button"
        title="参照…"
        aria-label="参照"
        disabled={disabled || browseDisabled}
        onClick={onBrowse}
      >
        <FolderIcon />
      </button>
    </div>
  );
}

/** Collapsible inspector section with an optional enable badge. */
export function Section({
  name,
  open,
  onToggle,
  badge,
  badgeState,
  muted = false,
  children,
}: {
  name: string;
  open: boolean;
  onToggle: () => void;
  badge?: string;
  badgeState?: "on" | "off";
  muted?: boolean;
  children: ReactNode;
}) {
  return (
    <div className={`sec ${open ? "is-open" : ""}`}>
      <button type="button" className="sec__head" onClick={onToggle}>
        <ChevronIcon className="sec__chevron" />
        <span className="sec__name">{name}</span>
        {badge && (
          <span className={`sec__badge ${badgeState ? `is-${badgeState}` : ""}`}>
            {badge}
          </span>
        )}
      </button>
      {open && (
        <div className={`sec__body ${muted ? "is-muted" : ""}`}>{children}</div>
      )}
    </div>
  );
}
