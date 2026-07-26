const W = 300;
const H = 100;

/** Throughput trace for the viewer — a job's fps history, scope-style. */
export function Scope({
  samples,
  label,
  unit,
}: {
  samples: number[];
  label: string;
  unit: string;
}) {
  const peak = samples.length > 0 ? Math.max(...samples) : 0;
  const top = peak > 0 ? peak * 1.15 : 1;
  const current = samples.at(-1) ?? null;

  const points = samples.map((value, index) => {
    const x = samples.length === 1 ? W : (index / (samples.length - 1)) * W;
    const y = H - (value / top) * H;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });

  return (
    <div className="scope">
      <div className="scope__head">
        <span>{label}</span>
        <b>
          {current === null ? "no signal" : current.toFixed(2)}
          {current !== null && <em> {unit}</em>}
        </b>
      </div>
      <svg
        className="scope__plot"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
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
            />
            <polyline
              className="scope__line"
              points={points.join(" ")}
              vectorEffect="non-scaling-stroke"
            />
          </>
        )}
      </svg>
      {points.length < 2 && <div className="scope__idle">no signal</div>}
      <div className="scope__axis">
        <span>{peak > 0 ? `peak ${peak.toFixed(1)}` : "—"}</span>
        <span>{samples.length > 0 ? `${samples.length} samples` : "待機中"}</span>
      </div>
    </div>
  );
}
