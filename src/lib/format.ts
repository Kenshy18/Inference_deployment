export function filename(value: string): string {
  return value.split(/[\\/]/).filter(Boolean).at(-1) ?? value;
}

export function duration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) {
    return "--:--";
  }
  const safe = Math.max(0, Math.floor(seconds));
  const h = Math.floor(safe / 3_600);
  const m = Math.floor((safe % 3_600) / 60);
  const s = safe % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function clock(value: string | null): string {
  if (!value) {
    return "--:--:--";
  }
  const date = new Date(value);
  return [date.getHours(), date.getMinutes(), date.getSeconds()]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

export function count(value: number): string {
  return value.toLocaleString("en-US");
}

export function rate(value: number | null, digits = 1): string | null {
  return value === null ? null : value.toFixed(digits);
}
