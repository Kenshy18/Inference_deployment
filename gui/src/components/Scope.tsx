import { useId } from "react";

const W = 300;
const H = 100;

/** Compact, reusable telemetry trace for the central monitor. */
export function Scope({
  samples,
  label,
  unit,
  color = "#5e8bff",
  fixedMax,
  decimals = 1,
}: {
  samples: number[];
  label: string;
  unit: string;
  color?: string;
  fixedMax?: number;
  decimals?: number;
}) {
  const rawId = useId();
  const gradientId = `scope-fill-${rawId.replaceAll(":", "")}`;
  const peak = samples.length > 0 ? Math.max(...samples) : 0;
  const top = fixedMax ?? (peak > 0 ? peak * 1.15 : 1);
  const current = samples.at(-1) ?? null;

  const points = samples.map((value, index) => {
    const x = samples.length === 1 ? W : (index / (samples.length - 1)) * W;
    const bounded = Math.min(top, Math.max(0, value));
    const y = H - (bounded / top) * H;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });

  return (
    <div className="scope">
      <div className="scope__head">
        <span>{label}</span>
        <b>
          {current === null ? "no signal" : current.toFixed(decimals)}
          {current !== null && <em> {unit}</em>}
        </b>
      </div>
      <svg
        className="scope__plot"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        aria-label={`${label} history`}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0.015" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((line) => (
          <line
            key={line}
            x1={0}
            x2={W}
            y1={H * line}
            y2={H * line}
            className="scope__grid"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {points.length > 1 && (
          <>
            <polygon
              className="scope__area"
              points={`0,${H} ${points.join(" ")} ${W},${H}`}
              fill={`url(#${gradientId})`}
            />
            <polyline
              className="scope__line"
              points={points.join(" ")}
              stroke={color}
              vectorEffect="non-scaling-stroke"
            />
          </>
        )}
      </svg>
      {points.length < 2 && <div className="scope__idle">no signal</div>}
      <div className="scope__axis">
        <span>
          {fixedMax
            ? `scale ${fixedMax}${unit}`
            : peak > 0
              ? `peak ${peak.toFixed(1)}`
              : "—"}
        </span>
        <span>{samples.length > 0 ? `${samples.length} pt` : "待機中"}</span>
      </div>
    </div>
  );
}
