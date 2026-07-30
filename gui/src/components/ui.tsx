import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { CheckIcon, ChevronIcon, FolderIcon } from "./Icons";

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

/** Custom dropdown. Chromium's native <select> opens an OS popup (GTK under
 *  WSLg) that ignores our theme and fails to dismiss on selection, so the
 *  list is rendered by React instead: it closes on select, outside pointer,
 *  scroll and Escape, and supports arrow-key navigation. */
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
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [anchor, setAnchor] = useState<{
    left: number;
    top: number;
    width: number;
    up: boolean;
  } | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const popRef = useRef<HTMLDivElement | null>(null);

  const current = options.find((option) => option.value === value);

  const openList = useCallback(() => {
    const trigger = buttonRef.current;
    if (!trigger) {
      return;
    }
    const rect = trigger.getBoundingClientRect();
    const estimated = Math.min(options.length * 26 + 10, 262);
    const up = rect.bottom + estimated > window.innerHeight - 8;
    setAnchor({ left: rect.left, top: up ? rect.top : rect.bottom, width: rect.width, up });
    setActive(
      Math.max(
        0,
        options.findIndex((option) => option.value === value),
      ),
    );
    setOpen(true);
  }, [options, value]);

  const commit = useCallback(
    (next: T) => {
      setOpen(false);
      if (next !== value) {
        onChange(next);
      }
      buttonRef.current?.focus();
    },
    [onChange, value],
  );

  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointerDown = (event: Event) => {
      const target = event.target as Node;
      if (
        popRef.current?.contains(target) ||
        buttonRef.current?.contains(target)
      ) {
        return;
      }
      setOpen(false);
    };
    const onScroll = (event: Event) => {
      if (popRef.current?.contains(event.target as Node)) {
        return;
      }
      setOpen(false);
    };
    const onResize = () => setOpen(false);
    window.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onResize);
    };
  }, [open]);

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (!open) {
      if (["Enter", " ", "ArrowDown", "ArrowUp"].includes(event.key)) {
        event.preventDefault();
        openList();
      }
      return;
    }
    if (event.key === "Escape" || event.key === "Tab") {
      setOpen(false);
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((index) => Math.min(options.length - 1, index + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((index) => Math.max(0, index - 1));
    } else if (event.key === "Home") {
      event.preventDefault();
      setActive(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setActive(options.length - 1);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const option = options[active];
      if (option) {
        commit(option.value);
      }
    }
  };

  return (
    <div className="select">
      <button
        ref={buttonRef}
        type="button"
        className={`select__btn ${open ? "is-open" : ""}`}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => (open ? setOpen(false) : openList())}
        onKeyDown={onKeyDown}
      >
        <span>{current?.label ?? ""}</span>
        <ChevronIcon className="select__chevron" />
      </button>
      {open &&
        anchor &&
        createPortal(
          <div
            ref={popRef}
            className="select__pop"
            role="listbox"
            style={{
              left: anchor.left,
              minWidth: anchor.width,
              top: anchor.up ? undefined : anchor.top + 4,
              bottom: anchor.up
                ? window.innerHeight - anchor.top + 4
                : undefined,
            }}
          >
            {options.map((option, index) => (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={option.value === value}
                className={[
                  option.value === value ? "is-selected" : "",
                  index === active ? "is-active" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                ref={(node) => {
                  if (node && index === active) {
                    node.scrollIntoView({ block: "nearest" });
                  }
                }}
                onPointerEnter={() => setActive(index)}
                onClick={() => commit(option.value)}
              >
                <span>{option.label}</span>
                {option.value === value && <CheckIcon />}
              </button>
            ))}
          </div>,
          document.body,
        )}
    </div>
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
      <button
        type="button"
        className="sec__head"
        aria-expanded={open}
        onClick={onToggle}
      >
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
