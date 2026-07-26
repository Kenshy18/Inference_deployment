import type { JobSnapshot, PipelineDraft } from "../../shared/types";
import { count, duration, rate } from "../lib/format";
import { modelSpec } from "../lib/models";
import { groupState, plannedStages, stageStates } from "../lib/stages";
import type { NodeState } from "../lib/stages";
import {
  CpuIcon,
  DatabaseIcon,
  EyeIcon,
  LayersIcon,
  VideoIcon,
} from "./Icons";
import { Scope } from "./Scope";
import { Panel } from "./ui";

function Metric({
  label,
  value,
  unit,
}: {
  label: string;
  value: string | null;
  unit?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <b
        className={value === null ? "is-null" : ""}
        title={value === null ? undefined : `${value}${unit ? ` ${unit}` : ""}`}
      >
        {value ?? "—"}
        {value !== null && unit && <em>{unit}</em>}
      </b>
    </div>
  );
}

function FlowNode({
  state,
  label,
  value,
  icon: Icon,
}: {
  state: NodeState;
  label: string;
  value: string;
  icon: typeof VideoIcon;
}) {
  return (
    <div className={`flow__node is-${state}`}>
      <Icon />
      <b>{label}</b>
      <span title={value}>{value}</span>
    </div>
  );
}

export function MonitorPanel({
  draft,
  job,
  elapsedSeconds,
  statusLabel,
  summary,
  fpsHistory,
}: {
  draft: PipelineDraft;
  job: JobSnapshot;
  elapsedSeconds: number;
  statusLabel: string;
  summary: string;
  fpsHistory: number[];
}) {
  const stages = plannedStages(draft);
  const states = stageStates(stages, job);
  const activeIndex = states.indexOf("active");
  const doneCount = states.filter((state) => state === "done").length;
  const stageFraction = job.telemetry.progress;

  const overall =
    job.status === "completed"
      ? 1
      : stages.length === 0 || (activeIndex === -1 && doneCount === 0)
        ? null
        : (doneCount + (activeIndex >= 0 ? (stageFraction ?? 0) : 0)) /
          stages.length;

  const remainingFrames =
    job.telemetry.totalFrames === null
      ? null
      : Math.max(0, job.telemetry.totalFrames - job.telemetry.processedFrames);
  const running = job.status === "running" || job.status === "cancelling";
  const eta =
    running && remainingFrames !== null && job.telemetry.fps
      ? remainingFrames / job.telemetry.fps
      : null;

  const overlayCount = [
    draft.overlay.raw,
    draft.overlay.tracked,
    draft.overlay.final,
    draft.overlay.faces,
  ].filter(Boolean).length;

  const nodes = [
    {
      key: "src",
      label: "Source",
      icon: VideoIcon,
      state: (draft.inputVideo ? "ready" : "waiting") as NodeState,
      value: draft.inputVideo ? "video" : "未選択",
    },
    {
      key: "inf",
      label: "Inference",
      icon: CpuIcon,
      state: groupState("inference", stages, states),
      value: draft.inference.enabled
        ? draft.inference.mode === "face"
          ? "face only"
          : modelSpec(draft.inference.segmentationModel).label
        : "既存 SQLite",
    },
    {
      key: "post",
      label: "Postprocess",
      icon: LayersIcon,
      state: groupState("postprocess", stages, states),
      value: draft.postprocess.enabled ? draft.postprocess.shapeMode : "skip",
    },
    {
      key: "ovl",
      label: "Overlay",
      icon: EyeIcon,
      state: groupState("overlay", stages, states),
      value:
        draft.overlay.enabled && overlayCount > 0
          ? `${overlayCount} proxy`
          : "skip",
    },
    {
      key: "out",
      label: "Handoff",
      icon: DatabaseIcon,
      state: (job.status === "completed" ? "done" : "waiting") as NodeState,
      value: "video + sqlite",
    },
  ];

  return (
    <Panel
      title="Monitor"
      meta={draft.inference.mode}
      actions={
        <span className="panel__meta">
          {job.dryRun ? "DRY RUN" : job.id ? `JOB ${job.id.slice(0, 19)}` : ""}
        </span>
      }
    >
      <div className="viewer">
        <div className="viewer__hud">
          <span>
            {stages.length > 0
              ? `${stages.length} STAGE${stages.length > 1 ? "S" : ""}`
              : "NO STAGE"}
          </span>
          <span className="is-right">
            {statusLabel.toUpperCase()} · {job.stage ?? "—"}
          </span>
        </div>

        <div className="viewer__stack">
          <div className="flow">
            {nodes.map((node, index) => (
              <div key={node.key} style={{ display: "contents" }}>
                {index > 0 && (
                  <span
                    className={`flow__link ${
                      node.state === "done"
                        ? "is-done"
                        : node.state === "active"
                          ? "is-active"
                          : ""
                    }`}
                  />
                )}
                <FlowNode
                  state={node.state}
                  label={node.label}
                  value={node.value}
                  icon={node.icon}
                />
              </div>
            ))}
          </div>

          <Scope samples={fpsHistory} label="throughput" unit="fps" />

          <div className="viewer__footer">
            <div className="readout">
              <b className={overall === null ? "is-idle" : ""}>
                {overall === null ? "—" : `${Math.round(overall * 100)}%`}
              </b>
              <i>{summary}</i>
            </div>
            <div className="eta">
              <b>{duration(elapsedSeconds)}</b>
              <span>elapsed{eta !== null ? ` · 残り ${duration(eta)}` : ""}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="timeline">
        {stages.length === 0 ? (
          <div className="timeline__cell">
            <b>実行するステージがありません</b>
          </div>
        ) : (
          stages.map((stage, index) => {
            const state = states[index];
            const indeterminate = state === "active" && stageFraction === null;
            return (
              <div
                key={stage.id}
                className={`timeline__cell is-${state} ${
                  indeterminate ? "is-indeterminate" : ""
                }`}
                title={stage.id}
              >
                <span
                  className="timeline__fill"
                  style={
                    state === "active" && !indeterminate
                      ? { width: `${(stageFraction ?? 0) * 100}%` }
                      : undefined
                  }
                />
                <b>{stage.label}</b>
              </div>
            );
          })
        )}
      </div>

      <div className="metrics">
        <Metric label="wall fps" value={rate(job.telemetry.fps, 2)} />
        <Metric label="compute" value={rate(job.telemetry.computeFps, 2)} unit="img/s" />
        <Metric
          label="frames"
          value={
            job.telemetry.processedFrames === 0 && job.telemetry.totalFrames === null
              ? null
              : `${count(job.telemetry.processedFrames)}${
                  job.telemetry.totalFrames
                    ? ` / ${count(job.telemetry.totalFrames)}`
                    : ""
                }`
          }
        />
        <Metric
          label="detections"
          value={job.telemetry.detections ? count(job.telemetry.detections) : null}
        />
        <Metric
          label="masks"
          value={job.telemetry.masks ? count(job.telemetry.masks) : null}
        />
        <Metric
          label="faces"
          value={job.telemetry.faces ? count(job.telemetry.faces) : null}
        />
        <Metric label="device" value={draft.inference.device || null} />
        <Metric
          label="codec"
          value={draft.overlay.enabled ? draft.overlay.codec : null}
        />
      </div>
    </Panel>
  );
}
